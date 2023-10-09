# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

import os, sys
import glob
import math
import multiprocessing
import numpy as np
import pandas as pd
import openslide
import csv
import time
from openslide import OpenSlideError
import PIL
from PIL import Image, ImageDraw, ImageFont
import datetime
import skimage.morphology as sk_morphology
from enum import Enum
from tqdm import tqdm
import argparse
from Kindai.codes.kindai import preprocessimg, annotate_label, generate_tile_annotation_summaries

## SETTINGS ##
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

MODEL_PATH1 = model_dir + '/NASNet_Large_pattern_augumented_all.h5'
MODEL_PATH2 = model_dir + '/NASNet_Large_tils_augumented_all.h5'
pattern_lists = ['MT','PG','SP','Stroma','UD']

#MODEL_PATH1 = model_dir + '/NASNet_Large_pattern_augumented.h5'
#MODEL_PATH2 = model_dir + '/NASNet_Large_tils_augumented_all.h5'
#pattern_lists = ['MT','PG','SP','Stroma']

parser = argparse.ArgumentParser()
## Parameters for image processing
parser.add_argument('--START', type=int, default=1, help='Slide number to start processing.')
parser.add_argument('--END', type=int, default=None, help='Slide number to end processing.')
parser.add_argument('--SCALE_FACTOR', type=int, default=32, help='Reduction factor of the training and filtered image. Set value ≥ 32')
parser.add_argument('--ROW_TILE_SIZE', type=int, default=331, help='Row pixel size of the tile.')
parser.add_argument('--COL_TILE_SIZE', type=int, default=331, help='Column pixel size of the tile.')
parser.add_argument('--TISSUE_THRESH', type=int, default=80, help='The thresholds of tissue amount at displaying tile.')
parser.add_argument('--EXPORT_TILE_THRESH', type=int, default=80, help='The thresholds of tissue amount at exporting tile.')
parser.add_argument('--EXPORT_TILE_SCALE', type=int, default=4, help='Reduction factor of the exporting tile.')
parser.add_argument('--DISPLAY_TILE_SUMMARY_LABELS', default=False, help='Set label on tile summary image.', action='store_true')
parser.add_argument('--singleprocess', default=False, help='Set single processing for task.', action='store_true')
parser.add_argument('--project', type=str, default='TCGA', help='Name of the project: Used as an prefix for images.')
parser.add_argument('--file_dir', type=str, default=None, help='Specify directory of the whole slide images.')
parser.add_argument('--working_dir', type=str, default=None, help='Specify working directory.')
parser.add_argument('--SRC_TRAIN_EXT', type=str, default='svs', help='File extension of input whole slide image.')
parser.add_argument('--num_process', type=int, default=16, help='Set process numbers low if error occurs in multiprocessing.')
parser.add_argument('--tiles_annotation', default=False, help='Predict labels of the tiles.', action='store_true')
parser.add_argument('--annotation_tile_size', type=int, default=10, help='Define annotation heatmap tile size.')

args = parser.parse_args()

# Directory and filepath settings
BASE_DIR = args.working_dir
# Directory of the input whole slide images
SRC_TRAIN_DIR = args.file_dir
# Project name: Used as an prefix for export image names
TRAIN_PREFIX = args.project + '_'

## EXPORT SETTINGS
# Save training image
SAVE_TRAININGIMAGE = True
# Save filter images
SAVE_FILTERDIMAGE = True
# Save tile summary images
SAVE_SUMMARY = True
# Save tiles
SAVE_TILES = False

## 出力ファイル先設定
DEST_TRAIN_SUFFIX = ""  # Example: "train-"
DEST_TRAIN_EXT = "png"
DEST_TRAIN_DIR = os.path.join(BASE_DIR, "training_" + DEST_TRAIN_EXT)
THUMBNAIL_SIZE = 300
THUMBNAIL_EXT = "jpg"
DEST_TRAIN_THUMBNAIL_DIR = os.path.join(BASE_DIR, "training_thumbnail_" + THUMBNAIL_EXT)
FILTER_SUFFIX = ""  # Example: "filter-"
FILTER_RESULT_TEXT = "filtered"
FILTER_DIR = os.path.join(BASE_DIR, "filter_" + DEST_TRAIN_EXT)
FILTER_THUMBNAIL_DIR = os.path.join(BASE_DIR, "filter_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_DIR = os.path.join(BASE_DIR, "tile_summary_" + DEST_TRAIN_EXT)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_" + DEST_TRAIN_EXT)
TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_thumbnail_" + THUMBNAIL_EXT)
DEST_TILE_EXT = "jpg"
TILE_DIR = os.path.join(BASE_DIR, "tiles")
TILE_SUFFIX = "tile"
DEST_DATAFRAME_DIR = os.path.join(BASE_DIR, "dataframe")

## Summaryファイルの色、フォント設定
HIGH_COLOR = (0, 255, 0)
LOW_COLOR = (255, 0, 0)
TILE_LABEL_TEXT_SIZE = 10
TILE_BORDER_SIZE = 1
FONT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
FONT_PATH = os.path.join(FONT_BASE_DIR, "Arial Bold.ttf")
SUMMARY_TITLE_FONT_PATH = os.path.join(FONT_BASE_DIR, "Courier New Bold.ttf")
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4


def open_slide(filename):
    """
    Open a whole-slide image (*.svs,.ndpi, etc).
    Args:
      filename: Name of the slide file.
    Returns:
      An OpenSlide object representing a whole-slide image.
    """
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide


def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
      filename: Name of the image file.
    returns:
      A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image


def open_image_np(filename):
    """
    Open an image (*.jpg, *.png, etc) as an RGB NumPy array.
    Args:
      filename: Name of the image file.
    returns:
      A NumPy representing an RGB image.
    """
    pil_img = open_image(filename)
    np_img = pil_to_np_rgb(pil_img)
    return np_img


def get_num_training_slides():
    """
    Obtain the total number of WSI training slide images.
    Returns:
      The total number of WSI training slide images.
    """
    num_training_slides = len(glob.glob(os.path.join(SRC_TRAIN_DIR, "*." + args.SRC_TRAIN_EXT)))
    return num_training_slides


def get_training_image_path(slide_number, dimension):
    """
    Convert slide number and optional dimensions to a training image path. If no dimensions are supplied,
    the corresponding file based on the slide number will be looked up in the file system using a wildcard.
    Example:
      5 -> ../images/training_png/HGSOC_005-32x-49920x108288-1560x3384.png
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the image file.
    """
    os.makedirs(DEST_TRAIN_DIR, exist_ok=True)
    padded_sl_num = str(slide_number).zfill(4)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(DEST_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        args.SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
        large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + DEST_TRAIN_EXT)
    return img_path


def get_training_thumbnail_path(slide_number, dimension):
    """
    Convert slide number and optional dimensions to a training thumbnail path. If no dimensions are
    supplied, the corresponding file based on the slide number will be looked up in the file system using a wildcard.
    Example:
      5 -> ../images/training_thumbnail_jpg/HGSOC_005-32x-49920x108288-1560x3384.jpg
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the thumbnail file.
    """
    os.makedirs(DEST_TRAIN_THUMBNAIL_DIR, exist_ok=True)
    padded_sl_num = str(slide_number).zfill(4)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(DEST_TRAIN_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        args.SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
        large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + THUMBNAIL_EXT)
    return img_path


def get_filter_image_path(slide_number, dimension):
    """
    Convert slide number to the path to the file that is the final result of filtering.
    Example:
      5 -> ../images/filter_png/HGSOC_005-32x-49920x108288-1560x3384-filtered.png
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the filter image file.
    """
    padded_sl_num = str(slide_number).zfill(4)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(FILTER_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        args.SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
        small_h) + "-" + FILTER_RESULT_TEXT + "." + DEST_TRAIN_EXT)
    return img_path


def get_filter_thumbnail_path(slide_number, dimension):
    """
    Convert slide number to the path to the file that is the final thumbnail result of filtering.
    Example:
      5 -> ../images/filter_thumbnail_jpg/HGSOC_005-32x-49920x108288-1560x3384-filtered.jpg
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the filter thumbnail file.
    """
    padded_sl_num = str(slide_number).zfill(4)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(FILTER_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        args.SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
        small_h) + "-" + FILTER_RESULT_TEXT + "." + THUMBNAIL_EXT)
    return img_path


def get_tile_summary_image_filename(slide_number, dimension, thumbnail=False):
    """
    Convert slide number to a tile summary image file name.
    Example:
      5, False -> HGSOC_005-tile_summary.png
      5, True -> HGSOC_005-tile_summary.jpg
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
      thumbnail: If True, produce thumbnail filename.
    Returns:
      The tile summary image file name.
    """
    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT
    padded_sl_num = str(slide_number).zfill(4)
    large_w, large_h, small_w, small_h = dimension
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(args.SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
        large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_SUMMARY_SUFFIX + "." + ext
    return img_filename


def get_tile_summary_image_path(slide_number, dimension):
    """
    Convert slide number to a path to a tile summary image file.
    Example:
      5 -> ../images/tile_summary_png/HGSOC_005-tile_summary.png
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the tile summary image file.
    """
    os.makedirs(TILE_SUMMARY_DIR, exist_ok=True)
    img_path = os.path.join(TILE_SUMMARY_DIR, get_tile_summary_image_filename(slide_number, dimension))
    return img_path


def get_tile_summary_thumbnail_path(slide_number, dimension):
    """
    Convert slide number to a path to a tile summary thumbnail file.
    Example:
      5 -> ../images/tile_summary_thumbnail_jpg/HGSOC_005-tile_summary.jpg
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the tile summary thumbnail file.
    """
    os.makedirs(TILE_SUMMARY_THUMBNAIL_DIR, exist_ok=True)
    img_path = os.path.join(TILE_SUMMARY_THUMBNAIL_DIR,
                            get_tile_summary_image_filename(slide_number, dimension, thumbnail=True))
    return img_path


def get_tile_summary_on_original_image_path(slide_number, dimension):
    """
    Convert slide number to a path to a tile summary on original image file.
    Example:
      5 -> ../images/tile_summary_on_original_png/HGSOC_005-tile_summary.png
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the tile summary on original image file.
    """
    os.makedirs(TILE_SUMMARY_ON_ORIGINAL_DIR, exist_ok=True)
    img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_DIR, get_tile_summary_image_filename(slide_number, dimension))
    return img_path


def get_tile_summary_on_original_thumbnail_path(slide_number, dimension):
    """
    Convert slide number to a path to a tile summary on original thumbnail file.
    Example:
      5 -> ../images/tile_summary_on_original_thumbnail_jpg/HGSOC_005-tile_summary.jpg
    Args:
      slide_number: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      Path to the tile summary on original thumbnail file.
    """
    os.makedirs(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR, exist_ok=True)
    img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR,
                            get_tile_summary_image_filename(slide_number, dimension, thumbnail=True))
    return img_path


def save_thumbnail(pil_img, size, path):
    """
    Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.
    Args:
      pil_img: The PIL image to save as a thumbnail.
      size:  The maximum width or height of the thumbnail.
      path: The path to the thumbnail.
    """
    max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
    img = pil_img.resize(max_size, PIL.Image.BILINEAR)
    thumbnail_dir = os.path.dirname(path)
    if thumbnail_dir != '' and not os.path.exists(thumbnail_dir):
        os.makedirs(thumbnail_dir)
    img.save(path)


def get_tile_image_dir(slide_num):
    """
    Obtain tile image dir based on tile information.
    Args:
      slide_num: slide number.
    Returns:
      Tile image directory.
    """
    padded_sl_num = str(slide_num).zfill(4)
    tile_dir = os.path.join(TILE_DIR, padded_sl_num)
    if SAVE_TILES:
        os.makedirs(tile_dir, exist_ok=True)
    return tile_dir


def get_tile_image_filename(slide_num, r, c):
  """
  Obtain tile image path based on tile information such as row, column.
  Args:
    slide_num: slide number.
    r: tile row number.
    c: tile column number.
  Returns:
    Path to image tile.
  """
  padded_sl_num = str(slide_num).zfill(4)
  tile_filename = padded_sl_num + "/" + TRAIN_PREFIX + padded_sl_num + "-" + \
                  TILE_SUFFIX + "-r%dc%d" % (r, c) + "." + DEST_TILE_EXT
  return tile_filename


def slide_to_scaled_pil_image(slide_number):
    """
    Convert a WSI training slide to a scaled-down PIL image.
    Args:
      slide_number: The slide number.
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    f = Filename()
    slide_filepath = f.filenames(slide_number)["filepath"]
    slide_name = f.filenames(slide_number)["filename"]
    print("Opening Slide #%d: %s" % (slide_number, slide_name))
    slide = open_slide(slide_filepath)
    large_w, large_h = slide.dimensions
    try:
        microns_per_pixel = float(slide.properties["openslide.mpp-x"])
    except:
        microns_per_pixel = 10 / float(slide.properties["openslide.objective-power"])
    new_w = math.floor(large_w / args.SCALE_FACTOR)
    new_h = math.floor(large_h / args.SCALE_FACTOR)

    if args.SRC_TRAIN_EXT == "svs" and (large_w > 65535) or (large_h > 65535): 
        # level should be set to 3 to avoid slide errors
        level = 3
    else:
        level = slide.get_best_level_for_downsample(args.SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    scaled_img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return scaled_img, large_w, large_h, new_w, new_h, microns_per_pixel


def training_slide_to_image(slide_number, save=SAVE_TRAININGIMAGE):
    """
    Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
    Args:
      slide_number: The slide number.
      save: if True, produce training image.
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    img, large_w, large_h, new_w, new_h, microns_per_pixel = slide_to_scaled_pil_image(slide_number)
    dimension = large_w, large_h, new_w, new_h

    if save:
        img_path = get_training_image_path(slide_number, dimension)
        img.save(img_path)

        thumbnail_path = get_training_thumbnail_path(slide_number, dimension)
        save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)

    np_img = pil_to_np_rgb(img)
    return np_img, dimension, microns_per_pixel


def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
      pil_img: The PIL Image.
    Returns:
      The PIL image converted to a NumPy array.
    """
    rgb = np.asarray(pil_img)
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    Args:
      np_img: The image represented as a NumPy array.
    Returns:
       The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.
    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    result = rgb * np.dstack([mask, mask, mask])
    return result


def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    Args:
      np_img: Image as a NumPy array.
    Returns:
      The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    Args:
      np_img: Image as a NumPy array.
    Returns:
      The percentage of the NumPy array that is tissue.
    """
    return 100 - mask_percent(np_img)


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.
    Args:
      np_img: Image as a NumPy array of type bool.
      min_size: Minimum size of small object to remove.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array (bool, float, or uint8).
    """
    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size // 2
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255
    return np_img


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.
    Args:
      np_img: RGB image as a NumPy array.
      green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """
    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255
    return np_img


def filter_grays(np_img, tolerance=15, output_type="bool"):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    Args:
      np_img: RGB image as a NumPy array.
      tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    np_img = np_img.astype('int')
    rg_diff = abs(np_img[:, :, 0] - np_img[:, :, 1]) <= tolerance
    rb_diff = abs(np_img[:, :, 0] - np_img[:, :, 2]) <= tolerance
    gb_diff = abs(np_img[:, :, 1] - np_img[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def apply_image_filters(np_img):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images.
    Args:
      np_img: Image as NumPy array.
    Returns:
      Resulting filtered image as a NumPy array.
    """
    rgb = np_img
    mask_not_green = filter_green_channel(rgb)
    mask_not_gray = filter_grays(rgb)
    mask_gray_green = mask_not_gray & mask_not_green
    mask_remove_small = filter_remove_small_objects(mask_gray_green, min_size=500, output_type="bool")
    rgb_remove_small = mask_rgb(rgb, mask_remove_small)
    img = rgb_remove_small
    return img


def apply_filters_to_image(slide_num, np_img, dimension, save=SAVE_FILTERDIMAGE):
    """
    Apply a set of filters to an image and optionally save and/or display filtered images.
    Args:
      slide_num: The slide number.
      np_img : Image as NumPy array.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
      save: If True, save filtered images.
    Returns:
      Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
      (used for HTML page generation).
    """
    if save and not os.path.exists(FILTER_DIR):
        os.makedirs(FILTER_DIR)
    filtered_np_img = apply_image_filters(np_img)

    if save:
        result_path = get_filter_image_path(slide_num, dimension)
        pil_img = np_to_pil(filtered_np_img)
        pil_img.save(result_path)
        thumbnail_path = get_filter_thumbnail_path(slide_num, dimension)
        save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_path)
    return filtered_np_img


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
    a column tile size.
    Args:
      rows: Number of rows.
      cols: Number of columns.
      row_tile_size: Number of pixels in a tile row.
      col_tile_size: Number of pixels in a tile column.
    Returns:
      Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
      into given the row tile size and the column tile size.
    """
    num_row_tiles = math.ceil(rows / row_tile_size)
    num_col_tiles = math.ceil(cols / col_tile_size)
    return num_row_tiles, num_col_tiles


def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain a list of tile coordinates
    (starting row, ending row, starting column, ending column, row number, column number).
    Args:
      rows: Number of rows.
      cols: Number of columns.
      row_tile_size: Number of pixels in a tile row.
      col_tile_size: Number of pixels in a tile column.
    Returns:
      List of tuples representing tile coordinates consisting of starting row, ending row,
      starting column, ending column, row number, column number.
    """
    indices = list()
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    for r in range(0, num_row_tiles):
        start_r = r * row_tile_size
        end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
        for c in range(0, num_col_tiles):
            start_c = c * col_tile_size
            end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
            indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
    return indices


def small_to_large_mapping(orig_tile_col, orig_tile_row, scaled_tile_col, scaled_tile_row, small_x, small_y):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    Args:
      small_pixel: The scaled-down width and height.
      large_dimensions: The width and height of the original whole-slide image.
    Returns:
      Tuple consisting of the scaled-up width and height.
    """
    large_x = round((orig_tile_col/args.SCALE_FACTOR/scaled_tile_col)*(args.SCALE_FACTOR * small_x))
    large_y = round((orig_tile_row/args.SCALE_FACTOR/scaled_tile_row)*(args.SCALE_FACTOR * small_y))
    return large_x, large_y


def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
    """
    Create a PIL summary image including top title area and right side and bottom padding.
    Args:
      np_img: Image as a NumPy array.
      title_area_height: Height of the title area at the top of the summary image.
      row_tile_size: The tile size in rows.
      col_tile_size: The tile size in columns.
      num_row_tiles: The number of row tiles.
      num_col_tiles: The number of column tiles.
    Returns:
      Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
      potentially a top title area and right side and bottom padding.
    """
    r = row_tile_size * num_row_tiles + title_area_height
    c = col_tile_size * num_col_tiles
    summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
    # add gray edges so that tile text does not get cut off
    summary_img.fill(120)
    # color title area white
    summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
    summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
    summary = np_to_pil(summary_img)
    return summary


def generate_tile_summaries(tile_sum, np_orig, filterd_np_img, save_summary=SAVE_SUMMARY):
    """
    Generate summary images/thumbnails showing a 'heatmap' representation of the tissue segmentation of all tiles.
    Args:
      tile_sum: TileSummary object.
      np_orig: Image as a NumPy array of original image.
      filterd_np_img: Image as a NumPy array of filtered result image
      save_summary: If True, save tile summary images.
    """
    z = 300  # height of area at top of summary slide
    slide_num = tile_sum.slide_num
    rows = tile_sum.scaled_h
    cols = tile_sum.scaled_w
    row_tile_size = tile_sum.scaled_tile_h
    col_tile_size = tile_sum.scaled_tile_w
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    summary = create_summary_pil_img(filterd_np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw = ImageDraw.Draw(summary)
    summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw_orig = ImageDraw.Draw(summary_orig)

    for t in tile_sum.tiles:
        border_color = tile_border_color(t.tissue_percentage)
        tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
        tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

    summary_txt = summary_title(tile_sum) + "\n" + summary_stats(tile_sum)

    summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
    draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
    draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

    if args.DISPLAY_TILE_SUMMARY_LABELS:
        count = 0
        for t in tile_sum.tiles:
            count += 1
            label = "R%d\nC%d" % (t.r, t.c)
            font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
            # drop shadow behind text
            draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)
            draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)

            draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
            draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

    if save_summary:
        dimension = tile_sum.orig_w, tile_sum.orig_h, tile_sum.scaled_w, tile_sum.scaled_h
        save_tile_summary_image(summary, slide_num, dimension)
        save_tile_summary_on_original_image(summary_orig, slide_num, dimension)


def tile_border_color(tissue_percentage):
    """
    Obtain the corresponding tile border color for a particular tile tissue percentage.
    Args:
      tissue_percentage: The tile tissue percentage
    Returns:
      The tile border color corresponding to the tile tissue percentage.
    """
    if tissue_percentage >= args.TISSUE_THRESH:
        border_color = HIGH_COLOR
    else:
        border_color = LOW_COLOR
    return border_color


def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
    """
    Draw a border around a tile with width TILE_BORDER_SIZE.
    Args:
      draw: Draw object for drawing on PIL image.
      r_s: Row starting pixel.
      r_e: Row ending pixel.
      c_s: Column starting pixel.
      c_e: Column ending pixel.
      color: Color of the border.
      border_size: Width of tile border in pixels.
    """
    for x in range(0, border_size):
        draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)


def save_tile_summary_image(pil_img, slide_num, dimension):
    """
    Save a tile summary image and thumbnail to the file system.
    Args:
      pil_img: Image as a PIL Image.
      slide_num: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    """
    filepath = get_tile_summary_image_path(slide_num, dimension)
    pil_img.save(filepath)

    thumbnail_filepath = get_tile_summary_thumbnail_path(slide_num, dimension)
    save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)


def save_tile_summary_on_original_image(pil_img, slide_num, dimension):
    """
    Save a tile summary on original image and thumbnail to the file system.
    Args:
      pil_img: Image as a PIL Image.
      slide_num: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    """
    filepath = get_tile_summary_on_original_image_path(slide_num, dimension)
    pil_img.save(filepath)

    thumbnail_filepath = get_tile_summary_on_original_thumbnail_path(slide_num, dimension)
    save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)


def summary_title(tile_summary):
    """
    Obtain tile summary title.
    Args:
      tile_summary: TileSummary object.
    Returns:
       The tile summary title.
    """
    return "Slide %03d Tile Summary:" % tile_summary.slide_num


def summary_stats(tile_summary):
    """
    Obtain various stats about the slide tiles.
    Args:
      tile_summary: TileSummary object.
    Returns:
      Various stats about the slide tiles as a string.
    """
    return "Slide name : %s\n" % tile_summary.file_name + \
           "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
           "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
           "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
           "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
           "Total Tissue: %3.2f%%\n" % tile_summary.tissue_percentage + \
           "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
           "  %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
               tile_summary.high, tile_summary.high / tile_summary.count * 100, args.TISSUE_THRESH) + \
           "  %5d (%5.2f%%) tiles <%d%% tissue\n" % (
               tile_summary.low, tile_summary.low / tile_summary.count * 100, args.TISSUE_THRESH) + \
           "Export setting: threshold %d%%, compressed to 1/%dx\n" % (
               args.EXPORT_TILE_THRESH, args.EXPORT_TILE_SCALE) + \
           "> %5d (%5.2f%%) tiles exported" % (tile_summary.export, tile_summary.export / tile_summary.count * 100)


def tile_to_pil_tile(wsimage, o_c_s, o_c_e, o_r_s, o_r_e):
    """
    Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.
    Args:
      slide_num: The slide number.
      o_c_s: position at original image
      o_c_e:
      o_r_s:
      o_r_e:
    Return:
      Tile as a PIL image.
    """
    x, y = o_c_s, o_r_s
    w, h = o_c_e - o_c_s, o_r_e - o_r_s
    tile_region = wsimage.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    tile_img = tile_region.convert("RGB")
    return tile_img


def score_tile(tile_tissue_percent):
    """
    Score tile based on tissue percentage.
    Args:
      tile_tissue_percent: The percentage of the tile judged to be tissue.
    Returns: score factor.
    """
    score = tile_tissue_percent / 100
    return score


def tissue_quantity(tissue_percentage):
    """
    Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.
    Args:
      tissue_percentage: The tile tissue percentage.
    Returns:
      TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
    """
    if tissue_percentage >= args.TISSUE_THRESH:
        return TissueQuantity.HIGH
    else:
        return TissueQuantity.LOW


def tissue_export(tissue_percentage):
    """
    Obtain Export tile enum member (EXPORT or NOEXPORT) for corresponding tissue percentage.
    Args:
      tissue_percentage: The tile tissue percentage.
    Returns:
      TissueExport enum member (EXPORT or NOEXPORT).
    """
    if tissue_percentage >= args.EXPORT_TILE_THRESH:
        return TissueExport.EXPORT
    else:
        return TissueExport.NOEXPORT


def create_tilesummary(num_process, start_index, end_index, tile_indices, tile_sum, np_img, slide_filepath):
    time.sleep(1.0*(num_process-1))
    tile_num = start_index-1
    high = 0
    low = 0
    export = 0
    noexport = 0
    list_idxs, list_images, list_tile = list(), list(), list()
    ws_image = open_slide(slide_filepath)
    slide_num = tile_sum.slide_num
    scaled_tile_col = tile_sum.scaled_tile_w
    scaled_tile_row = tile_sum.scaled_tile_h
    orig_tile_col = tile_sum.orig_tile_w
    orig_tile_row = tile_sum.orig_tile_h
    microns_per_pixel = tile_sum.microns_per_pixel

    for idx in range(start_index, end_index+1):
        tile_num += 1
        r_s, r_e, c_s, c_e, r, c = tile_indices[idx-1]
        np_tile = np_img[r_s:r_e, c_s:c_e]
        t_p = tissue_percent(np_tile)
        amount = tissue_quantity(t_p)
        if amount == TissueQuantity.HIGH:
            high += 1
        elif amount == TissueQuantity.LOW:
            low += 1
        export_count = tissue_export(t_p)
        if export_count == TissueExport.EXPORT:
            export += 1
        elif export_count == TissueExport.NOEXPORT:
            noexport += 1
        o_c_s, o_r_s = small_to_large_mapping(orig_tile_col, orig_tile_row, scaled_tile_col, scaled_tile_row, c_s, r_s)
        o_c_e, o_r_e = small_to_large_mapping(orig_tile_col, orig_tile_row, scaled_tile_col, scaled_tile_row, c_e, r_e)
        if (o_c_e - o_c_s) > orig_tile_col:
            o_c_e -= 1
        if (o_r_e - o_r_s) > orig_tile_row:
            o_r_e -= 1
        t_p_score = score_tile(t_p)

        if (t_p >= args.EXPORT_TILE_THRESH) & ((c_e - c_s) == scaled_tile_col) & ((r_e - r_s) == scaled_tile_row):
            pil_scaled_tile = tile_to_pil_tile(ws_image, o_c_s, o_c_e, o_r_s, o_r_e)
            tile_filename = get_tile_image_filename(slide_num, r, c)
            np_processed_tile = preprocessimg(pil_scaled_tile, microns_per_pixel, args.EXPORT_TILE_SCALE,
                                              args.ROW_TILE_SIZE, args.COL_TILE_SIZE)
            if SAVE_TILES:
                tile_path = os.path.join(TILE_DIR, tile_filename)
                export_img = np_to_pil(np_processed_tile)
                export_img.save(tile_path)
            list_idxs.append(tile_num)
            list_images.append(np_processed_tile)
        else:
            tile_filename = None
        tile = Tile(tile_sum, slide_num, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s, o_c_e, t_p, t_p_score, tile_filename)
        list_tile.append(tile)
    del ws_image
    return list_idxs, list_images, list_tile, high, low, export, noexport

def score_tiles(slide_num, np_img, dimension, microns_per_pixel):
    """
    Score all tiles for a slide and return the results in a TileSummary object.
    Args:
      slide_num: The slide number.
      np_img: the NumPy tile image in the Tile objects.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    Returns:
      TileSummary object which includes a list of Tile objects containing information about each tile.
    """
    f = Filename()
    slide_filepath = f.filenames(slide_num)["filepath"]
    file_name = f.filenames(slide_num)["filename"]
    o_w, o_h, w, h = dimension
    orig_tile_col = int(16 * (((args.COL_TILE_SIZE*0.25/microns_per_pixel)//16)+1) * args.EXPORT_TILE_SCALE)
    orig_tile_row = int(16 * (((args.ROW_TILE_SIZE*0.25/microns_per_pixel)//16)+1) * args.EXPORT_TILE_SCALE)
    scaled_tile_col = int(orig_tile_col / args.SCALE_FACTOR)
    scaled_tile_row = int(orig_tile_row / args.SCALE_FACTOR)
    num_row_tiles, num_col_tiles = get_num_tiles(h, w, scaled_tile_row, scaled_tile_col)
    if SAVE_TILES:
        tile_dir = get_tile_image_dir(slide_num)
    else:
        tile_dir = None

    tile_sum = TileSummary(slide_num=slide_num,
                           file_name=file_name,
                           orig_w=o_w,
                           orig_h=o_h,
                           orig_tile_w=orig_tile_col,
                           orig_tile_h=orig_tile_row,
                           scaled_w=w,
                           scaled_h=h,
                           scaled_tile_w=scaled_tile_col,
                           scaled_tile_h=scaled_tile_row,
                           tissue_percentage=tissue_percent(np_img),
                           num_col_tiles=num_col_tiles,
                           num_row_tiles=num_row_tiles,
                           tile_dir=tile_dir,
                           microns_per_pixel=microns_per_pixel)

    tile_num = 0
    high = 0
    low = 0
    export = 0
    noexport = 0

    tile_indices = get_tile_indices(h, w, scaled_tile_row, scaled_tile_col)
    list_idxs, list_images = list(), list()

    if args.singleprocess:
        ws_image = open_slide(slide_filepath)
        with tqdm(total=len(tile_indices), desc="[Processing tiles in slide #%d]" % slide_num) as pbar:
            for t in tile_indices:
                tile_num += 1  # tile_num
                r_s, r_e, c_s, c_e, r, c = t
                np_tile = np_img[r_s:r_e, c_s:c_e]
                t_p = tissue_percent(np_tile)
                amount = tissue_quantity(t_p)
                if amount == TissueQuantity.HIGH:
                    high += 1
                elif amount == TissueQuantity.LOW:
                    low += 1
                export_count = tissue_export(t_p)
                if export_count == TissueExport.EXPORT:
                    export += 1
                elif export_count == TissueExport.NOEXPORT:
                    noexport += 1
                o_c_s, o_r_s = small_to_large_mapping(orig_tile_col, orig_tile_row, scaled_tile_col, scaled_tile_row, c_s, r_s)
                o_c_e, o_r_e = small_to_large_mapping(orig_tile_col, orig_tile_row, scaled_tile_col, scaled_tile_row, c_e, r_e)
                if (o_c_e - o_c_s) > orig_tile_col:
                    o_c_e -= 1
                if (o_r_e - o_r_s) > orig_tile_row:
                    o_r_e -= 1
                t_p_score = score_tile(t_p)
                if (t_p >= args.EXPORT_TILE_THRESH) & ((c_e - c_s) == scaled_tile_col) & ((r_e - r_s) == scaled_tile_row):
                    pil_scaled_tile = tile_to_pil_tile(ws_image, o_c_s, o_c_e, o_r_s, o_r_e) 
                    tile_filename = get_tile_image_filename(slide_num, r, c)
                    np_processed_tile = preprocessimg(pil_scaled_tile, microns_per_pixel, args.EXPORT_TILE_SCALE,
                                                      args.ROW_TILE_SIZE, args.COL_TILE_SIZE)
                    if SAVE_TILES:
                        tile_path = os.path.join(TILE_DIR, tile_filename)
                        export_img = np_to_pil(np_processed_tile)
                        export_img.save(tile_path)
                    list_idxs.append(tile_num)
                    list_images.append(np_processed_tile)
                else:
                    tile_filename = None
                tile = Tile(tile_sum, slide_num, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s, o_c_e, t_p, t_p_score, tile_filename)
                tile_sum.tiles.append(tile)
                pbar.update(1)

    else:
        num_processes = min(os.cpu_count(), args.num_process)
        tiles_per_process = len(tile_indices) / num_processes
        print("Number of processes: " + str(num_processes))
        tasks = []
        for num_process in range(1, num_processes + 1):
            start_index = (num_process - 1) * tiles_per_process + 1
            end_index = num_process * tiles_per_process
            start_index = int(start_index)
            end_index = int(end_index) 
            tasks.append((num_process, start_index, end_index, tile_indices, tile_sum, np_img, slide_filepath))
            #if start_index == end_index:
            #    print("Task #" + str(num_process) + ": Process tiles " + str(start_index))
            #else:
            #    print("Task #" + str(num_process) + ": Process tiles " + str(start_index) + " to " + str(end_index))
        # start tasks
        results = []
        bar_format = '{desc} {percentage: 3.0f}% | {bar:16}{r_bar}'
        with multiprocessing.Pool(num_processes) as pool, tqdm(total=len(tile_indices), desc="[Processing tiles in slide #%d]" % slide_num, bar_format=bar_format) as pbar:
            for t in tasks:
                result = pool.apply_async(create_tilesummary, t)
                results.append(result)
            for result in results:
                idxs, images, list_tile, high_count, low_count, export_count, noexport_count = result.get()
                high += high_count
                low += low_count
                export += export_count
                noexport += noexport_count
                list_idxs.extend(idxs)
                list_images.extend(images)
                for tile in list_tile:
                    tile_sum.tiles.append(tile)
                    pbar.update(1)

    tile_sum.count = len(tile_indices)
    tile_sum.high = high
    tile_sum.low = low
    tile_sum.export = export
    tile_sum.noexport = noexport
    return tile_sum, list_idxs, list_images


def summary_and_features_from_tiles(slide_num, save_summary=SAVE_SUMMARY):
    """
    Generate tile summary and export tiles for slide.
    Args:
      slide_num: The slide number.
      save_summary: If True, save tile summary images.
    """
    np_img, dimension, microns_per_pixel = training_slide_to_image(slide_num)
    fileterd_np_img = apply_filters_to_image(slide_num, np_img, dimension)
    tile_sum, list_idxs, list_images = score_tiles(slide_num, fileterd_np_img, dimension, microns_per_pixel)
    generate_tile_summaries(tile_sum, np_img, fileterd_np_img, save_summary=save_summary)
    return np_img, tile_sum, list_idxs, list_images


def process_range_to_tiles(start_index, end_index):
    """
    Generate tile summaries and tiles for a range of images.
    Args:
      start_index: Starting index (inclusive).
      end_index: Ending index (inclusive).
    """
    tile_sum_lists = list()
    for slide_num in range(start_index, end_index + 1):
        t = Time()
        np_orig, tile_summary, list_idxs, list_images = summary_and_features_from_tiles(slide_num)
        case_id = TRAIN_PREFIX + str(tile_summary.slide_num).zfill(4)
        df = pd.DataFrame.from_records([tile.to_dict() for tile in tile_summary.tiles])

        if args.tiles_annotation and len(list_images)>0:
            import silence_tensorflow.auto
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            from tensorflow.keras.models import load_model
            print('Loading models...')
            model1 = load_model(MODEL_PATH1, compile=False)
            model2 = load_model(MODEL_PATH2, compile=False)
            print('Loading models completed.')
            print('[Predicting %s tiles in slide #%s]'%(str(len(list_images)), str(tile_summary.slide_num)))
            pattern_label_lists, pattern_pred_lists = annotate_label(model1, list_images)
            tils_label_lists, tils_pred_lists = annotate_label(model2, list_images)
            pattern_label_lists = [pattern_lists[k] for k in pattern_label_lists]
            df_predictions = pd.DataFrame(columns=['tile_num', 'pattern_label', 'pattern_prob', 'tils_label','tils_prob'])
            df_predictions['tile_num'] = list_idxs
            df_predictions['pattern_label'] = pattern_label_lists
            df_predictions['pattern_prob'] = pattern_pred_lists
            df_predictions['tils_label'] = tils_label_lists
            df_predictions['tils_prob'] = tils_pred_lists
            df_predictions.loc[df_predictions['pattern_label'] == 'Stroma', 'tils_label'] = None
            df_predictions.loc[df_predictions['pattern_label'] == 'Stroma', 'tils_prob'] = None
            df = pd.merge(df, df_predictions, on='tile_num', how='left')
            df_annotation = df[df['tile_filename'].notnull()]
            row_tile_size, col_tile_size = tile_summary.scaled_tile_h, tile_summary.scaled_tile_w
            num_row_tiles, num_col_tiles = tile_summary.num_row_tiles, tile_summary.num_col_tiles
            summary_orig = create_summary_pil_img(np_orig, 0, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
            annotation_img = np.zeros([num_row_tiles*args.annotation_tile_size, num_col_tiles*args.annotation_tile_size, 3], dtype=np.uint8)
            annotation_pattern_img, annotation_pattern_path = generate_tile_annotation_summaries(df_annotation, np_to_pil(annotation_img), args.annotation_tile_size, case_id, BASE_DIR, mode='pattern')
            annotation_pattern_img = Image.blend(summary_orig, annotation_pattern_img.resize(summary_orig.size), 0.3)
            annotation_pattern_img.save(annotation_pattern_path)
            annotation_tils_img, annotation_tils_path = generate_tile_annotation_summaries(df_annotation, np_to_pil(annotation_img), args.annotation_tile_size, case_id, BASE_DIR, mode='tils')
            annotation_tils_img = Image.blend(summary_orig, annotation_tils_img.resize(summary_orig.size), 0.3)
            annotation_tils_img.save(annotation_tils_path)
            del summary_orig, model1, model2, annotation_pattern_img, annotation_tils_img, df_predictions, df_annotation
        df.to_csv(os.path.join(DEST_DATAFRAME_DIR, case_id + '.csv'), index=False)
        tile_sum_list = tile_summary.to_dict()
        tile_sum_lists.append(tile_sum_list)
        print("Done slide processing: %s | Time: %-14s" % (case_id, str(t.elapsed())))
        del np_orig, tile_summary, list_images, df
    return tile_sum_lists


class Filename:
    """
    Class for collecting file name.
    """
    def __init__(self):
        root_dir = SRC_TRAIN_DIR
        self.image_names = sorted(glob.glob(os.path.join(root_dir, '*')))

    def filenames(self, idx):
        file_name = os.path.splitext(os.path.basename(self.image_names[idx - 1]))[0]
        file_path = self.image_names[idx - 1]
        filenames = {'filename': file_name, 'filepath': file_path}
        return filenames


class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed


class TileSummary:
    """
    Class for tile summary information.
    """

    slide_num = None
    orig_w = None
    orig_h = None
    orig_tile_w = None
    orig_tile_h = None
    scale_factor = args.SCALE_FACTOR
    scaled_w = None
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None
    microns_per_pixel = None

    count = 0
    high = 0
    low = 0

    def __init__(self, slide_num, file_name, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h,
                 scaled_tile_w, scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles, tile_dir,
                 microns_per_pixel):
        self.slide_num = slide_num
        self.file_name = file_name
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        self.tissue_percentage = tissue_percentage
        self.num_col_tiles = num_col_tiles
        self.num_row_tiles = num_row_tiles
        self.tile_dir = tile_dir
        self.microns_per_pixel = microns_per_pixel
        self.tiles = []

    def __str__(self):
        return summary_title(self) + "\n" + summary_stats(self)

    def to_dict(self):
        return {
            'ID': TRAIN_PREFIX + str(self.slide_num).zfill(4),
            'filename': self.file_name,
            'orig_w': self.orig_w,
            'orig_h': self.orig_h,
            'orig_tile_w': self.orig_tile_w,
            'orig_tile_h': self.orig_tile_h,
            'scaled_w': self.scaled_w,
            'scaled_h': self.scaled_h,
            'scaled_tile_w': self.scaled_tile_w,
            'scaled_tile_h': self.scaled_tile_h,
            'tissue_percentage': self.tissue_percentage,
            'num_col_tiles': self.num_col_tiles,
            'num_row_tiles': self.num_row_tiles,
            'tile_dir': self.tile_dir,
            'microns_per_pixel': self.microns_per_pixel
        }


class Tile:
    """
    Class for information about a tile.
    """
    def __init__(self, tile_summary, slide_num, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e,
                 o_c_s, o_c_e, tissue_percentage, t_p_score, tile_filename):
        self.tile_summary = tile_summary
        self.slide_num = slide_num
        self.tile_num = tile_num
        self.r = r
        self.c = c
        self.r_s = r_s
        self.r_e = r_e
        self.c_s = c_s
        self.c_e = c_e
        self.o_r_s = o_r_s
        self.o_r_e = o_r_e
        self.o_c_s = o_c_s
        self.o_c_e = o_c_e
        self.tissue_percentage = tissue_percentage
        self.t_p_score = t_p_score
        self.tile_filename = tile_filename

    def __str__(self):
        return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
            self.tile_num, self.r, self.c, self.tissue_percentage, self.t_p_score)

    def __repr__(self):
        return "\n" + self.__str__()

    def tissue_quantity(self):
        return tissue_quantity(self.tissue_percentage)

    def tissue_export(self):
        return tissue_export(self.tissue_percentage)

    def to_dict(self):
        return {
            'ID': TRAIN_PREFIX + str(self.tile_summary.slide_num).zfill(4),
            'filename': self.tile_summary.file_name,
            'orig_w': self.tile_summary.orig_w,
            'orig_h': self.tile_summary.orig_h,
            'orig_tile_w': self.tile_summary.orig_tile_w,
            'orig_tile_h': self.tile_summary.orig_tile_h,
            'scaled_w': self.tile_summary.scaled_w,
            'scaled_h': self.tile_summary.scaled_h,
            'scaled_tile_w': self.tile_summary.scaled_tile_w,
            'scaled_tile_h': self.tile_summary.scaled_tile_h,
            'tissue_percentage': self.tile_summary.tissue_percentage,
            'num_col_tiles': self.tile_summary.num_col_tiles,
            'num_row_tiles': self.tile_summary.num_row_tiles,
            'tile_num': self.tile_num,
            'row': self.r,
            'col': self.c,
            'row_start': self.r_s,
            'row_end': self.r_e,
            'column_start': self.c_s,
            'column_end': self.c_e,
            'orig_row_start': self.o_r_s,
            'orig_row_end': self.o_r_e,
            'orig_column_start': self.o_c_s,
            'orig_column_end': self.o_c_e,
            'tile_tissue_score': self.t_p_score,
            'tile_filename': self.tile_filename,
        }


class TissueQuantity(Enum):
    LOW = 0
    HIGH = 1


class TissueExport(Enum):
    EXPORT = 0
    NOEXPORT = 1


def main():
    os.makedirs(args.working_dir, exist_ok=True)
    assert os.path.isdir(args.working_dir), print('Specify working directory')
    os.makedirs(DEST_DATAFRAME_DIR, exist_ok=True)
    timer = Time()
    # start tasks
    start_index = args.START
    if args.END == None:
        end_index = get_num_training_slides()
    else: 
        end_index = args.END
    tile_sum_lists = process_range_to_tiles(start_index, end_index)
    tile_sum_info = pd.DataFrame.from_records(tile_sum_lists)
    tile_sum_info.to_csv(os.path.join(DEST_DATAFRAME_DIR, 'summary_%s_%s.csv'%(str(start_index), str(end_index))), index=False)
    print("Time to generate tile annotation summaries: %s\n" % str(timer.elapsed()))
    
    # Notification of process completion
    import slackweb
    slack = slackweb.Slack(url='https://hooks.slack.com/services/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')
    slack.notify(text="Completed %s projects: %s" % (args.project, str(timer.elapsed())))

if __name__ == '__main__':
    main()
