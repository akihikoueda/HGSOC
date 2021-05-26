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

import os
import glob
import math
import multiprocessing
import numpy as np
import pandas as pd
import openslide
from openslide import OpenSlideError
import PIL
from PIL import Image, ImageDraw, ImageFont
import datetime
import skimage.morphology as sk_morphology
from enum import Enum
from keras.models import Model
from keras.applications import NASNetLarge

## SETTINGS ##
## Parameters for image processing
# FILTER処理を行う画像の縮小倍率 (例 SCALE_FACTOR = 32 → 縦,横 1/32倍画像を用いて画像処理)
SCALE_FACTOR = 32
# 分割する際の1枚あたりのタイルサイズ
ROW_TILE_SIZE = 662
COL_TILE_SIZE = 662
# タイル分割時の組織量に基づく表示閾値
TISSUE_HIGH_THRESH = 50
TISSUE_LOW_THRESH = 10
# タイルを書き出す際の閾値と縮小倍率 (例 EXPORT_SCALE = 2 → 縦,横 1/2倍でタイル書き出し)
EXPORT_TILE_THRESH = 80
EXPORT_TILE_SCALE = 2
# Input image size for feature extraction (depending on the models)
input_size = 331
# Tileサマリ画像上でのラベル有無設定
DISPLAY_TILE_SUMMARY_LABELS = True

## BASIC SETTINGS
# Specify base directory
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
# Directory of the input whole slide images
SRC_TRAIN_DIR = os.path.join(BASE_DIR, "original")
# File extension of input whole slide image (e.g. svs, ndpi)
SRC_TRAIN_EXT = "svs"
# Project name: Used as an prefix for export image names
TRAIN_PREFIX = "HGSOC_"
# File extension of output images (e.g. png, jpg, tif)
DEST_TILE_EXT = "jpg"

## EXPORT SETTINGS
# Save training image
SAVE_TRAININGIMAGE = True
# Save filter images
SAVE_FILTERDIMAGE = True
# Save tile summary images
SAVE_SUMMARY = True
# Feature extraction
CREATE_FEATURES = True

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
TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TILE_EXT)
TILE_SUFFIX = "tile"
DEST_DataFrame_DIR = os.path.join(BASE_DIR, "dataframe")

## Summaryファイルの色、フォント設定
HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)
TILE_LABEL_TEXT_SIZE = 10
TILE_BORDER_SIZE = 2
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
    num_training_slides = len(glob.glob1(SRC_TRAIN_DIR, "*." + SRC_TRAIN_EXT))
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
    padded_sl_num = str(slide_number).zfill(3)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(DEST_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
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
    padded_sl_num = str(slide_number).zfill(3)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(DEST_TRAIN_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
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
    padded_sl_num = str(slide_number).zfill(3)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(FILTER_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
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
    padded_sl_num = str(slide_number).zfill(3)
    large_w, large_h, small_w, small_h = dimension
    img_path = os.path.join(FILTER_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
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
    padded_sl_num = str(slide_number).zfill(3)
    large_w, large_h, small_w, small_h = dimension
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
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
    if not os.path.exists(TILE_SUMMARY_DIR):
        os.makedirs(TILE_SUMMARY_DIR)
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
    if not os.path.exists(TILE_SUMMARY_THUMBNAIL_DIR):
        os.makedirs(TILE_SUMMARY_THUMBNAIL_DIR)
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
    if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_DIR):
        os.makedirs(TILE_SUMMARY_ON_ORIGINAL_DIR)
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
    if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR):
        os.makedirs(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR)
    img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR,
                            get_tile_summary_image_filename(slide_number, dimension, thumbnail=True))
    return img_path


def save_thumbnail(pil_img, size, path, display_path=False):
    """
    Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.
    Args:
      pil_img: The PIL image to save as a thumbnail.
      size:  The maximum width or height of the thumbnail.
      path: The path to the thumbnail.
      display_path: If True, display thumbnail path in console.
    """
    max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
    img = pil_img.resize(max_size, PIL.Image.BILINEAR)
    if display_path:
        print("Saving thumbnail to: " + path)
    thumbnail_dir = os.path.dirname(path)
    if thumbnail_dir != '' and not os.path.exists(thumbnail_dir):
        os.makedirs(thumbnail_dir)
    img.save(path)


def slide_to_scaled_pil_image(slide_number):
    """
    Convert a WSI training slide to a scaled-down PIL image.
    Args:
      slide_number: The slide number.
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """

    def _get_concat_h(img_lst):
        width, height, h = sum([img.width for img in img_lst]), img_lst[0].height, 0
        dst = Image.new('RGB', (width, height))
        for img in img_lst:
            dst.paste(img, (h, 0))
            h += img.width
        return dst

    def _get_concat_v(img_lst):
        width, height, v = img_lst[0].width, sum([img.height for img in img_lst]), 0
        dst = Image.new('RGB', (width, height))
        for img in img_lst:
            dst.paste(img, (0, v))
            v += img.height
        return dst

    f = Filename()
    slide_filepath = f.filenames(slide_number)["filepath"]
    slide_name = f.filenames(slide_number)["filename"]
    print("Opening Slide #%d: %s" % (slide_number, slide_name))
    slide = open_slide(slide_filepath)
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)

    if SRC_TRAIN_EXT == "svs" and large_w > 65535 or large_h > 65535:
        unit_x, unit_y = 1024, 1024
        downsample_rate = slide.level_downsamples[level]
        w_rep, h_rep = int(large_w / unit_x) + 1, int(large_h / unit_y) + 1
        w_end, h_end = large_w % unit_x, large_h % unit_y
        w_size, h_size = unit_x, unit_y
        w_start, h_start = 0, 0
        v_lst = []
        for i in range(h_rep):
            if i == h_rep - 1:
                h_size = h_end
            h_lst = []
            for j in range(w_rep):
                if j == w_rep - 1:
                    w_size = w_end
                tile_img = slide.read_region((w_start, h_start), level, (
                    math.floor(w_size / downsample_rate), math.floor(h_size / downsample_rate)))
                tile_img = tile_img.convert("RGB")
                h_lst.append(tile_img)
                w_start += unit_x
            v_lst.append(h_lst)
            w_size = unit_x
            h_start += unit_y
            w_start = 0
        concat_h = [_get_concat_h(v) for v in v_lst]
        concat_hv = _get_concat_v(concat_h)
        scaled_img = concat_hv.resize((new_w, new_h), PIL.Image.BILINEAR)
        return scaled_img, large_w, large_h, new_w, new_h
    else:
        whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
        whole_slide_image = whole_slide_image.convert("RGB")
        scaled_img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        return scaled_img, large_w, large_h, new_w, new_h


def training_slide_to_image(slide_number, save=SAVE_TRAININGIMAGE):
    """
    Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
    Args:
      slide_number: The slide number.
      save: if True, produce training image.
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number)
    dimension = large_w, large_h, new_w, new_h
    if save:
        img_path = get_training_image_path(slide_number, dimension)
        print("Saving image to: " + img_path)
        if not os.path.exists(DEST_TRAIN_DIR):
            os.makedirs(DEST_TRAIN_DIR)
        img.save(img_path)
        thumbnail_path = get_training_thumbnail_path(slide_number, dimension)
        save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)
    np_img = pil_to_np_rgb(img)
    return np_img, dimension


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
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    np_info(result, "Mask RGB", t.elapsed())
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
    t = Time()
    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size // 2
        print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
            mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Remove Small Objs", t.elapsed())
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
    t = Time()
    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d"
            % (mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Filter Green Channel", t.elapsed())
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
    t = Time()
    np_img = np_img.astype(np.int)
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
    np_info(result, "Filter Grays", t.elapsed())
    return result


def np_info(np_arr, name=None, elapsed=None):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.
    Args:
      np_arr: The NumPy array.
      name: The (optional) name of the array.
      elapsed: The (optional) time elapsed to perform a filtering operation.
    """
    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))


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
    t = Time()
    print("Processing slide #%d" % slide_num)
    if save and not os.path.exists(FILTER_DIR):
        os.makedirs(FILTER_DIR)
    filtered_np_img = apply_image_filters(np_img)

    if save:
        t1 = Time()
        result_path = get_filter_image_path(slide_num, dimension)
        pil_img = np_to_pil(filtered_np_img)
        pil_img.save(result_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

        t1 = Time()
        thumbnail_path = get_filter_thumbnail_path(slide_num, dimension)
        save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

    print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

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


def small_to_large_mapping(small_pixel, large_dimensions):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    Args:
      small_pixel: The scaled-down width and height.
      large_dimensions: The width and height of the original whole-slide image.
    Returns:
      Tuple consisting of the scaled-up width and height.
    """
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
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
    z = 350  # height of area at top of summary slide
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

    if DISPLAY_TILE_SUMMARY_LABELS:
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
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        border_color = HIGH_COLOR
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        border_color = MEDIUM_COLOR
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        border_color = LOW_COLOR
    else:
        border_color = NONE_COLOR
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
    t = Time()
    filepath = get_tile_summary_image_path(slide_num, dimension)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum", str(t.elapsed()), filepath))

    t = Time()
    thumbnail_filepath = get_tile_summary_thumbnail_path(slide_num, dimension)
    save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Thumb", str(t.elapsed()), thumbnail_filepath))


def save_tile_summary_on_original_image(pil_img, slide_num, dimension):
    """
    Save a tile summary on original image and thumbnail to the file system.
    Args:
      pil_img: Image as a PIL Image.
      slide_num: The slide number.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
    """
    t = Time()
    filepath = get_tile_summary_on_original_image_path(slide_num, dimension)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig", str(t.elapsed()), filepath))

    t = Time()
    thumbnail_filepath = get_tile_summary_on_original_thumbnail_path(slide_num, dimension)
    save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
    print(
        "%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig T", str(t.elapsed()), thumbnail_filepath))


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
           "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
           "Total Tissue: %3.2f%%\n" % tile_summary.tissue_percentage + \
           "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
           "  %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
               tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
           "  %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
               tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
               TISSUE_HIGH_THRESH) + \
           "  %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
               tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
           "  %5d (%5.2f%%) tiles =0%% tissue\n" % (tile_summary.none, tile_summary.none / tile_summary.count * 100) + \
           "Export setting: threshold %d%%, compressed to %dx%d px (1/%dx)\n" % (
               EXPORT_TILE_THRESH, ROW_TILE_SIZE / EXPORT_TILE_SCALE,
               COL_TILE_SIZE / EXPORT_TILE_SCALE, EXPORT_TILE_SCALE) + \
           "> %5d (%5.2f%%) tiles exported" % (tile_summary.export, tile_summary.export / tile_summary.count * 100)


def normalizestaining(np_img, Io=240, alpha=1, beta=0.15):
    """
    Normalize staining appearence of H&E stained images
    Args:
      np_img: RGB input image
      Io:    (optional) transmitted light intensity
      beta:  (optional) OD threshold for transparent pixels (default: 0.15);
      alpha: (optional) tolerance for the pseudo-min and pseudo-max (default: 1);
    Return:
      Inorm: normalized image
      H: hematoxylin image
      E: eosin image
    """
    img = np.array(np_img)
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    # define height and width of image
    h, w, c = img.shape
    # reshape image
    img = img.reshape((-1, 3))
    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / Io)
    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    # eigvecs *= -1
    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)
    pil_Inorm = Image.fromarray(np.uint8(Inorm))
    return pil_Inorm


def tile_to_pil_tile(slide_num, o_c_s, o_c_e, o_r_s, o_r_e):
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
    f = Filename()
    slide_filepath = f.filenames(slide_num)["filepath"]
    s = open_slide(slide_filepath)

    x, y = o_c_s, o_r_s
    w, h = o_c_e - o_c_s, o_r_e - o_r_s
    tile_w = math.floor(w / EXPORT_TILE_SCALE)
    tile_h = math.floor(h / EXPORT_TILE_SCALE)
    tile_region = s.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    tile_img = tile_region.convert("RGB")
    pil_img = tile_img.resize((tile_w, tile_h), PIL.Image.BILINEAR)
    pil_normalized_img = normalizestaining(pil_img)
    return pil_normalized_img


def score_tiles(slide_num, np_img, dimension, create_features=CREATE_FEATURES):
    """
    Score all tiles for a slide and return the results in a TileSummary object.
    Args:
      slide_num: The slide number.
      np_img: the NumPy tile image in the Tile objects.
      dimension: Tuple containing large_w, large_h, small_w, and small_h.
      create_features: if True, add features extracted from CNN models
    Returns:
      TileSummary object which includes a list of Tile objects containing information about each tile.
    """
    f = Filename()
    file_name = f.filenames(slide_num)["filename"]
    o_w, o_h, w, h = dimension
    row_tile_size = round(ROW_TILE_SIZE / SCALE_FACTOR)
    col_tile_size = round(COL_TILE_SIZE / SCALE_FACTOR)
    num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

    tile_sum = TileSummary(slide_num=slide_num,
                           file_name=file_name,
                           orig_w=o_w,
                           orig_h=o_h,
                           orig_tile_w=COL_TILE_SIZE,
                           orig_tile_h=ROW_TILE_SIZE,
                           scaled_w=w,
                           scaled_h=h,
                           scaled_tile_w=col_tile_size,
                           scaled_tile_h=row_tile_size,
                           tissue_percentage=tissue_percent(np_img),
                           num_col_tiles=num_col_tiles,
                           num_row_tiles=num_row_tiles)

    if create_features:
        model = NASNetLarge(include_top=True, weights="imagenet", input_tensor=None, input_shape=None)
        layer_name = model.layers[-2].name
        intermediante_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0
    export = 0
    noexport = 0

    tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
    pbar_info = "[Processing tiles in slide #%d]" % (slide_num)
    with tqdm(total=len(tile_indices), desc=pbar_info, position=slide_num) as pbar:
        for t in tile_indices:
            pbar.update(1)
            count += 1  # tile_num
            r_s, r_e, c_s, c_e, r, c = t
            np_tile = np_img[r_s:r_e, c_s:c_e]
            t_p = tissue_percent(np_tile)
            amount = tissue_quantity(t_p)
            if amount == TissueQuantity.HIGH:
                high += 1
            elif amount == TissueQuantity.MEDIUM:
                medium += 1
            elif amount == TissueQuantity.LOW:
                low += 1
            elif amount == TissueQuantity.NONE:
                none += 1
            export_count = tissue_export(t_p)
            if export_count == TissueExport.EXPORT:
                export += 1
            elif export_count == TissueExport.NOEXPORT:
                noexport += 1
            o_c_s, o_r_s = small_to_large_mapping((c_s, r_s), (o_w, o_h))
            o_c_e, o_r_e = small_to_large_mapping((c_e, r_e), (o_w, o_h))
            # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
            if (o_c_e - o_c_s) > COL_TILE_SIZE:
                o_c_e -= 1
            if (o_r_e - o_r_s) > ROW_TILE_SIZE:
                o_r_e -= 1
            t_p_score = score_tile(t_p)
            if t_p >= EXPORT_TILE_THRESH:
                pil_scaled_tile = tile_to_pil_tile(slide_num, o_c_s, o_c_e, o_r_s, o_r_e)
                w, h = pil_scaled_tile.size
                pil_scaled_tile = pil_scaled_tile.resize((input_size, input_size), Image.LANCZOS)
                np_scaled_tile = np.array(pil_scaled_tile).astype("float32") / 255
                if w == h:
                    if create_features:
                        image = np_scaled_tile
                        image = image[None, ...]
                        features = intermediante_layer_model.predict(image)[0]
            else:
                np_scaled_tile = None
                if create_features:
                    features = None

            tile = Tile(tile_sum, slide_num, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                        o_c_e, t_p, t_p_score, np_scaled_tile, features)
            tile_sum.tiles.append(tile)

    tile_sum.count = count
    tile_sum.high = high
    tile_sum.medium = medium
    tile_sum.low = low
    tile_sum.none = none
    tile_sum.export = export
    tile_sum.noexport = noexport
    return tile_sum


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
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        return TissueQuantity.HIGH
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        return TissueQuantity.MEDIUM
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        return TissueQuantity.LOW
    else:
        return TissueQuantity.NONE


def tissue_export(tissue_percentage):
    """
    Obtain Export tile enum member (EXPORT or NOEXPORT) for corresponding tissue percentage.
    Args:
      tissue_percentage: The tile tissue percentage.
    Returns:
      TissueExport enum member (EXPORT or NOEXPORT).
    """
    if tissue_percentage >= EXPORT_TILE_THRESH:
        return TissueExport.EXPORT
    else:
        return TissueExport.NOEXPORT


def summary_and_features_from_tiles(slide_num, save_summary=SAVE_SUMMARY):
    """
    Generate tile summary and export tiles for slide.
    Args:
      slide_num: The slide number.
      save_summary: If True, save tile summary images.
    """
    np_img, dimension = training_slide_to_image(slide_num)
    fileterd_np_img = apply_filters_to_image(slide_num, np_img, dimension)
    tile_sum = score_tiles(slide_num, fileterd_np_img, dimension)
    generate_tile_summaries(tile_sum, np_img, fileterd_np_img, save_summary=save_summary)
    return tile_sum


def process_list_to_tiles(image_num_list):
    """
    Generate tile summaries and tiles for a list of images.
    Args:
      image_num_list: List of image numbers.
    """
    if os.path.exists(DEST_DataFrame_DIR):
        os.makedirs(DEST_DataFrame_DIR)
    for slide_num in image_num_list:
        t = Time()
        tile_summary = summary_and_features_from_tiles(slide_num)
        df_name = TRAIN_PREFIX + str(tile_summary.slide_num).zfill(3)
        df = pd.DataFrame.from_records([tile.to_dict() for tile in tile_summary.tiles])
        df.to_pickle(os.path.join(DEST_DataFrame_DIR, df_name + ".pkl"))
        print("Done slide processing: %s | Time: %-14s" % (df_name, str(t.elapsed())))
        del df, tile_summary


def process_range_to_tiles(start_index, end_index):
    """
    Generate tile summaries and tiles for a range of images.
    Args:
      start_index: Starting index (inclusive).
      end_index: Ending index (inclusive).
    """
    if os.path.exists(DEST_DataFrame_DIR):
        os.makedirs(DEST_DataFrame_DIR)
    for slide_num in range(start_index, end_index + 1):
        t = Time()
        tile_summary = summary_and_features_from_tiles(slide_num)
        df_name = TRAIN_PREFIX + str(tile_summary.slide_num).zfill(3)
        df = pd.DataFrame.from_records([tile.to_dict() for tile in tile_summary.tiles])
        df.to_pickle(os.path.join(DEST_DataFrame_DIR, df_name + ".pkl"))
        print("Done slide processing: %s | Time: %-14s" % (df_name, str(t.elapsed())))
        del df, tile_summary


def singleprocess_filter_and_features(start_index, image_num_list=None):
    """
    Generate tile summaries and tiles for training images using a single process.
    Args:
      start_index: index number of the slide to start processing
      # save_summary: If True, save tile summary images.
      # save_data: If True, save tile data to csv file.
      # html: If True, generate HTML page to display tiled images
      image_num_list: Optionally specify a list of image slide numbers.
    """
    t = Time()
    print("Generating tile annotation summaries\n")

    if image_num_list is not None:
        process_list_to_tiles(image_num_list)
    else:
        num_training_slides = get_num_training_slides()
        process_range_to_tiles(start_index, start_index + num_training_slides - 1)
    print("Time to generate tile annotation summaries: %s\n" % str(t.elapsed()))


def multiprocess_filter_and_features(image_num_list=None):
    """
    Generate tile summaries and tiles for all training images using multiple processes (one process per core).
    Args:
      # save_summary: If True, save tile summary images.
      # save_data: If True, save tile data to csv file.
      # html: If True, generate HTML page to display tiled images.
      image_num_list: Optionally specify a list of image slide numbers.
    """
    timer = Time()
    print("Generating tile summaries (multiprocess)\n")

    num_processes = os.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    if image_num_list is not None:
        num_train_images = len(image_num_list)
    else:
        num_train_images = get_num_training_slides()
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    print("Number of processes: " + str(num_processes))
    print("Number of training images: " + str(num_train_images))

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        if image_num_list is not None:
            sublist = image_num_list[start_index - 1:end_index]
            tasks.append(sublist)
            print("Task #" + str(num_process) + ": Process slides " + str(sublist))
        else:
            tasks.append((start_index, end_index))
            if start_index == end_index:
                print("Task #" + str(num_process) + ": Process slide " + str(start_index))
            else:
                print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        if image_num_list is not None:
            results.append(pool.apply_async(process_list_to_tiles, t))
        else:
            results.append(pool.apply_async(process_range_to_tiles, t))
    for result in results:
        result.get()
    print("Time to generate tile previews (multiprocess): %s\n" % str(timer.elapsed()))


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
    scale_factor = SCALE_FACTOR
    scaled_w = None
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0

    def __init__(self, slide_num, file_name, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h,
                 scaled_tile_w, scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
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
        self.tiles = []

    def __str__(self):
        return summary_title(self) + "\n" + summary_stats(self)


class Tile:
    """
    Class for information about a tile.
    """
    def __init__(self, tile_summary, slide_num, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e,
                 o_c_s, o_c_e, tissue_percentage, t_p_score, np_scaled_tile, features):
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
        self.np_scaled_tile = np_scaled_tile
        self.features = features

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
            'ID': TRAIN_PREFIX + str(self.tile_summary.slide_num).zfill(3),
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
            'row_start': self.r_s,
            'row_end': self.r_e,
            'column_start': self.c_s,
            'column_end': self.c_e,
            'orig_row_start': self.o_r_s,
            'orig_row_end': self.o_r_e,
            'orig_column_start': self.o_c_s,
            'orig_column_end': self.o_c_e,
            'tile_tissue_score': self.t_p_score,
            'np_img': self.np_scaled_tile,
            'features': self.features,
        }


class TissueQuantity(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class TissueExport(Enum):
    EXPORT = 0
    NOEXPORT = 1


## 実行コード
if __name__ == '__main__':
    # multiprocessing.freeze_support()  # windows上での実行の場合のみON
    multiprocess_filter_and_features()

    ## 途中の番号より実行する場合
    # start_ind = 1  # 開始番号
    # singleprocess_filter_and_features(start_ind)
