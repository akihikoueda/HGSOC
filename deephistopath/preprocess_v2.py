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

import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlideError
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import re
import sys
import datetime
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
import colorsys
from enum import Enum

## 画像処理設定項目：
# FILTER処理を行う画像の縮小倍率 (例 2 → 縦1/2, 横 1/2で書き出し)
SCALE_FACTOR = 32
# 分割する際の1枚あたりのタイルサイズ
ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
# タイル分割時の組織量に基づく表示閾値
TISSUE_HIGH_THRESH = 50
TISSUE_LOW_THRESH = 10
# タイルを書き出す際の閾値と縮小倍率 (例 2 → 縦1/2, 横 1/2で書き出し)
EXPORT_TILE_THRESH = 80
EXPORT_TILE_SCALE = 2
# Tileサマリ画像上でのラベル有無設定
DISPLAY_TILE_SUMMARY_LABELS = True
# HE image normalization parameters
normalize_Io = 240
normalize_alpha = 1
normalize_beta = 0.15

## 入出力ファイル設定項目:
# BASE DIRECTORYを記入
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"images")
# Whole Slide Image 入力ファイルフォルダ名をSRC_TRAIN_DIRで規定
SRC_TRAIN_DIR = os.path.join(BASE_DIR, "original")
# Whole Slide Image 入力ファイル名は TRAIN_PREFIX + 3桁の番号 で規定
TRAIN_PREFIX = "HGSOC_"
# Whole Slide Image 入力ファイル拡張子を SRC_TRAIN_EXT で規定 (svs, ndpi 等)
SRC_TRAIN_EXT = "ndpi"
# 出力TILEファイル拡張子を DEST_TILE_EXT で規定 (png, jpg, tif 等)
DEST_TILE_EXT = "jpg"

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
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True
FILTER_HTML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TILE_SUMMARY_DIR = os.path.join(BASE_DIR, "tile_summary_" + DEST_TRAIN_EXT)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_" + DEST_TRAIN_EXT)
TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True
TILE_SUMMARY_HTML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data")
TILE_DATA_SUFFIX = "tile_data"
TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TILE_EXT)
TILE_SUFFIX = "tile"

## Exportファイルの色設定
HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

TILE_LABEL_TEXT_SIZE = 10
TILE_BORDER_SIZE = 2

FONT_PATH = "/Library/Fonts/Arial Bold.ttf"
SUMMARY_TITLE_FONT_PATH = "/Library/Fonts/Courier New Bold.ttf"
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)

TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

## 画像処理 parser
# Slide.pyよりmerge

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


def get_training_slide_path(slide_number):
  """
  Convert slide number to a path to the corresponding WSI training slide file.
  Example:
    5 -> ../images/HGSOC_005.svs
  Args:
    slide_number: The slide number.
  Returns:
    Path to the WSI training slide file.
  """
  padded_sl_num = str(slide_number).zfill(3)
  slide_filepath = os.path.join(SRC_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "." + SRC_TRAIN_EXT)
  return slide_filepath


def get_tile_image_path(tile):
  """
  Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
  pixel width, and pixel height.
  Args:
    tile: Tile object.
  Returns:
    Path to image tile.
  """
  t = tile
  padded_sl_num = str(t.slide_num).zfill(3)
  tile_path = os.path.join(TILE_DIR, padded_sl_num,
                           TRAIN_PREFIX + padded_sl_num + "-" + TILE_SUFFIX + "-r%d-c%d-x%d-y%d-wh%d-e%d" % (
                             t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, math.floor((t.o_r_e - t.o_r_s)/EXPORT_TILE_SCALE)) + "." + DEST_TILE_EXT)
  return tile_path


def get_training_image_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):
  """
  Convert slide number and optional dimensions to a training image path. If no dimensions are supplied,
  the corresponding file based on the slide number will be looked up in the file system using a wildcard.
  Example:
    5 -> ../images/training_png/HGSOC_005-32x-49920x108288-1560x3384.png
  Args:
    slide_number: The slide number.
    large_w: Large image width.
    large_h: Large image height.
    small_w: Small image width.
    small_h: Small image height.
  Returns:
     Path to the image file.
  """
  padded_sl_num = str(slide_number).zfill(3)
  if large_w is None and large_h is None and small_w is None and small_h is None:
    wildcard_path = os.path.join(DEST_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "*." + DEST_TRAIN_EXT)
    img_path = glob.glob(wildcard_path)[0]
  else:
    img_path = os.path.join(DEST_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
      SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
      large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + DEST_TRAIN_EXT)
  return img_path


def get_training_thumbnail_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):
  """
  Convert slide number and optional dimensions to a training thumbnail path. If no dimensions are
  supplied, the corresponding file based on the slide number will be looked up in the file system using a wildcard.
  Example:
    5 -> ../images/training_thumbnail_jpg/HGSOC_005-32x-49920x108288-1560x3384.jpg
  Args:
    slide_number: The slide number.
    large_w: Large image width.
    large_h: Large image height.
    small_w: Small image width.
    small_h: Small image height.
  Returns:
     Path to the thumbnail file.
  """
  padded_sl_num = str(slide_number).zfill(3)
  if large_w is None and large_h is None and small_w is None and small_h is None:
    wilcard_path = os.path.join(DEST_TRAIN_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "*." + THUMBNAIL_EXT)
    img_path = glob.glob(wilcard_path)[0]
  else:
    img_path = os.path.join(DEST_TRAIN_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
      SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
      large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + THUMBNAIL_EXT)
  return img_path


def get_filter_image_path(slide_number, filter_number, filter_name_info):
  """
  Convert slide number, filter number, and text to a path to a filter image file.
  Example:
    5, 1, "rgb" -> ../images/filter_png/HGSOC_005-001-rgb.png
  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
  Returns:
    Path to the filter image file.
  """
  dir = FILTER_DIR
  if not os.path.exists(dir):
    os.makedirs(dir)
  img_path = os.path.join(dir, get_filter_image_filename(slide_number, filter_number, filter_name_info))
  return img_path


def get_filter_thumbnail_path(slide_number, filter_number, filter_name_info):
  """
  Convert slide number, filter number, and text to a path to a filter thumbnail file.
  Example:
    5, 1, "rgb" -> ../images/filter_thumbnail_jpg/HGSOC_005-001-rgb.jpg
  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
  Returns:
    Path to the filter thumbnail file.
  """
  dir = FILTER_THUMBNAIL_DIR
  if not os.path.exists(dir):
    os.makedirs(dir)
  img_path = os.path.join(dir, get_filter_image_filename(slide_number, filter_number, filter_name_info, thumbnail=True))
  return img_path


def get_filter_image_filename(slide_number, filter_number, filter_name_info, thumbnail=False):
  """
  Convert slide number, filter number, and text to a filter file name.
  Example:
    5, 1, "rgb", False -> HGSOC_005-001-rgb.png
    5, 1, "rgb", True -> HGSOC_005-001-rgb.jpg
  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
    thumbnail: If True, produce thumbnail filename.
  Returns:
    The filter image or thumbnail file name.
  """
  if thumbnail:
    ext = THUMBNAIL_EXT
  else:
    ext = DEST_TRAIN_EXT
  padded_sl_num = str(slide_number).zfill(3)
  padded_fi_num = str(filter_number).zfill(3)
  img_filename = TRAIN_PREFIX + padded_sl_num + "-" + padded_fi_num + "-" + FILTER_SUFFIX + filter_name_info + "." + ext
  return img_filename


def get_tile_summary_image_path(slide_number):
  """
  Convert slide number to a path to a tile summary image file.
  Example:
    5 -> ../images/tile_summary_png/HGSOC_005-tile_summary.png
  Args:
    slide_number: The slide number.
  Returns:
    Path to the tile summary image file.
  """
  if not os.path.exists(TILE_SUMMARY_DIR):
    os.makedirs(TILE_SUMMARY_DIR)
  img_path = os.path.join(TILE_SUMMARY_DIR, get_tile_summary_image_filename(slide_number))
  return img_path


def get_tile_summary_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a tile summary thumbnail file.
  Example:
    5 -> ../images/tile_summary_thumbnail_jpg/HGSOC_005-tile_summary.jpg
  Args:
    slide_number: The slide number.
  Returns:
    Path to the tile summary thumbnail file.
  """
  if not os.path.exists(TILE_SUMMARY_THUMBNAIL_DIR):
    os.makedirs(TILE_SUMMARY_THUMBNAIL_DIR)
  img_path = os.path.join(TILE_SUMMARY_THUMBNAIL_DIR, get_tile_summary_image_filename(slide_number, thumbnail=True))
  return img_path


def get_tile_summary_on_original_image_path(slide_number):
  """
  Convert slide number to a path to a tile summary on original image file.
  Example:
    5 -> ../images/tile_summary_on_original_png/HGSOC_005-tile_summary.png
  Args:
    slide_number: The slide number.
  Returns:
    Path to the tile summary on original image file.
  """
  if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_DIR):
    os.makedirs(TILE_SUMMARY_ON_ORIGINAL_DIR)
  img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_DIR, get_tile_summary_image_filename(slide_number))
  return img_path


def get_tile_summary_on_original_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a tile summary on original thumbnail file.
  Example:
    5 -> ../images/tile_summary_on_original_thumbnail_jpg/HGSOC_005-tile_summary.jpg
  Args:
    slide_number: The slide number.
  Returns:
    Path to the tile summary on original thumbnail file.
  """
  if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR):
    os.makedirs(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR)
  img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR,
                          get_tile_summary_image_filename(slide_number, thumbnail=True))
  return img_path


def get_tile_summary_image_filename(slide_number, thumbnail=False):
  """
  Convert slide number to a tile summary image file name.
  Example:
    5, False -> HGSOC_005-tile_summary.png
    5, True -> HGSOC_005-tile_summary.jpg
  Args:
    slide_number: The slide number.
    thumbnail: If True, produce thumbnail filename.
  Returns:
    The tile summary image file name.
  """
  if thumbnail:
    ext = THUMBNAIL_EXT
  else:
    ext = DEST_TRAIN_EXT
  padded_sl_num = str(slide_number).zfill(3)

  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_SUMMARY_SUFFIX + "." + ext

  return img_filename


def get_tile_data_filename(slide_number):
  """
  Convert slide number to a tile data file name.
  Example:
    5 -> HGSOC_005-32x-49920x108288-1560x3384-tile_data.csv
  Args:
    slide_number: The slide number.
  Returns:
    The tile data file name.
  """
  padded_sl_num = str(slide_number).zfill(3)

  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  data_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_DATA_SUFFIX + ".csv"

  return data_filename


def get_tile_data_path(slide_number):
  """
  Convert slide number to a path to a tile data file.
  Example:
    5 -> ../images/tile_data/HGSOC_005-32x-49920x108288-1560x3384-tile_data.csv
  Args:
    slide_number: The slide number.
  Returns:
    Path to the tile data file.
  """
  if not os.path.exists(TILE_DATA_DIR):
    os.makedirs(TILE_DATA_DIR)
  file_path = os.path.join(TILE_DATA_DIR, get_tile_data_filename(slide_number))
  return file_path


def get_filter_image_result(slide_number):
  """
  Convert slide number to the path to the file that is the final result of filtering.
  Example:
    5 -> ../images/filter_png/HGSOC_005-32x-49920x108288-1560x3384-filtered.png
  Args:
    slide_number: The slide number.
  Returns:
    Path to the filter image file.
  """
  padded_sl_num = str(slide_number).zfill(3)
  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_path = os.path.join(FILTER_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
    SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
    small_h) + "-" + FILTER_RESULT_TEXT + "." + DEST_TRAIN_EXT)
  return img_path


def get_filter_thumbnail_result(slide_number):
  """
  Convert slide number to the path to the file that is the final thumbnail result of filtering.

  Example:
    5 -> ../images/filter_thumbnail_jpg/HGSOC_005-32x-49920x108288-1560x3384-filtered.jpg
  Args:
    slide_number: The slide number.
  Returns:
    Path to the filter thumbnail file.
  """
  padded_sl_num = str(slide_number).zfill(3)
  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_path = os.path.join(FILTER_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
    SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
    small_h) + "-" + FILTER_RESULT_TEXT + "." + THUMBNAIL_EXT)
  return img_path


def parse_dimensions_from_image_filename(filename):
  """
  Parse an image filename to extract the original width and height and the converted width and height.
  Example:
    "HGSOC_011-32x-97103x79079-3034x2471-tile_summary.png" -> (97103, 79079, 3034, 2471)
  Args:
    filename: The image filename.
  Returns:
    Tuple consisting of the original width, original height, the converted width, and the converted height.
  """
  m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
  large_w = int(m.group(1))
  large_h = int(m.group(2))
  small_w = int(m.group(3))
  small_h = int(m.group(4))
  return large_w, large_h, small_w, small_h


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


def training_slide_to_image(slide_number):
  """
  Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
  Args:
    slide_number: The slide number.
  """

  img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number)

  img_path = get_training_image_path(slide_number, large_w, large_h, new_w, new_h)
  print("Saving image to: " + img_path)
  if not os.path.exists(DEST_TRAIN_DIR):
    os.makedirs(DEST_TRAIN_DIR)
  img.save(img_path)

  thumbnail_path = get_training_thumbnail_path(slide_number, large_w, large_h, new_w, new_h)
  save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)


def slide_to_scaled_pil_image(slide_number):
  """
  Convert a WSI training slide to a scaled-down PIL image.
  Args:
    slide_number: The slide number.
  Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
  """
  slide_filepath = get_training_slide_path(slide_number)
  print("Opening Slide #%d: %s" % (slide_number, slide_filepath))
  slide = open_slide(slide_filepath)

  large_w, large_h = slide.dimensions
  new_w = math.floor(large_w / SCALE_FACTOR)
  new_h = math.floor(large_h / SCALE_FACTOR)
  level = slide.get_best_level_for_downsample(SCALE_FACTOR)
  whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
  whole_slide_image = whole_slide_image.convert("RGB")
  img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
  return img, large_w, large_h, new_w, new_h


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
  dir = os.path.dirname(path)
  if dir != '' and not os.path.exists(dir):
    os.makedirs(dir)
  img.save(path)


def get_num_training_slides():
  """
  Obtain the total number of WSI training slide images.
  Returns:
    The total number of WSI training slide images.
  """
  num_training_slides = len(glob.glob1(SRC_TRAIN_DIR, "*." + SRC_TRAIN_EXT))
  return num_training_slides


def training_slide_range_to_images(start_ind, end_ind):
  """
  Convert a range of WSI training slides to smaller images (in a format such as jpg or png).
  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
  Returns:
    The starting index and the ending index of the slides that were converted.
  """
  for slide_num in range(start_ind, end_ind + 1):
    training_slide_to_image(slide_num)
  return (start_ind, end_ind)


def singleprocess_training_slides_to_images():
  """
  Convert all WSI training slides to smaller images using a single process.
  """
  t = Time()

  num_train_images = get_num_training_slides()
  training_slide_range_to_images(1, num_train_images)

  t.elapsed_display()


def multiprocess_training_slides_to_images():
  """
  Convert all WSI training slides to smaller images using multiple processes (one process per core).
  Each process will process a range of slide numbers.
  """
  timer = Time()

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)

  num_train_images = get_num_training_slides()
  if num_processes > num_train_images:
    num_processes = num_train_images
  images_per_process = num_train_images / num_processes

  print("Number of processes: " + str(num_processes))
  print("Number of training images: " + str(num_train_images))

  # each task specifies a range of slides
  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * images_per_process + 1
    end_index = num_process * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    tasks.append((start_index, end_index))
    if start_index == end_index:
      print("Task #" + str(num_process) + ": Process slide " + str(start_index))
    else:
      print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    results.append(pool.apply_async(training_slide_range_to_images, t))

  for result in results:
    (start_ind, end_ind) = result.get()
    if start_ind == end_ind:
      print("Done converting slide %d" % start_ind)
    else:
      print("Done converting slides %d through %d" % (start_ind, end_ind))

  timer.elapsed_display()

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False

def pil_to_np_rgb(pil_img):
  """
  Convert a PIL Image to a NumPy array.
  Note that RGB PIL (w, h) -> NumPy (h, w, 3).
  Args:
    pil_img: The PIL Image.
  Returns:
    The PIL image converted to a NumPy array.
  """
  t = Time()
  rgb = np.asarray(pil_img)
  np_info(rgb, "RGB", t.elapsed())
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

  if ADDITIONAL_NP_STATS is False:
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
  else:
    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
      name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
  """
  Convert a NumPy array to a PIL image, add text to the image, and display the image.
  Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
  """
  result = np_to_pil(np_img)
  # if gray, convert to RGB for display
  if result.mode == 'L':
    result = result.convert('RGB')
  draw = ImageDraw.Draw(result)
  if text is not None:
    font = ImageFont.truetype(font_path, size)
    if bg:
      (x, y) = draw.textsize(text, font)
      draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
    draw.text((2, 0), text, color, font=font)
  result.show()


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

# Filter.pyよりmerge

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
    new_min_size = min_size / 2
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
      "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
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


def filter_grays(rgb, tolerance=15, output_type="bool"):
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
  (h, w, c) = rgb.shape

  rgb = rgb.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter Grays", t.elapsed())
  return result


def apply_image_filters(np_img, slide_num=None, info=None, save=False, display=False):
  """
  Apply filters to image as NumPy array and optionally save and/or display filtered images.
  Args:
    np_img: Image as NumPy array.
    slide_num: The slide number (used for saving/displaying).
    info: Dictionary of slide information (used for HTML display).
    save: If True, save image.
    display: If True, display image.
  Returns:
    Resulting filtered image as a NumPy array.
  """
  rgb = np_img
  save_display(save, display, info, rgb, slide_num, 1, "Original", "rgb")

  mask_not_green = filter_green_channel(rgb)
  rgb_not_green = mask_rgb(rgb, mask_not_green)
  save_display(save, display, info, rgb_not_green, slide_num, 2, "Not Green", "rgb-not-green")

  mask_not_gray = filter_grays(rgb)
  rgb_not_gray = mask_rgb(rgb, mask_not_gray)
  save_display(save, display, info, rgb_not_gray, slide_num, 3, "Not Gray", "rgb-not-gray")

  mask_gray_green = mask_not_gray & mask_not_green
  rgb_gray_green = mask_rgb(rgb, mask_gray_green)
  save_display(save, display, info, rgb_gray_green, slide_num, 7, "Not Gray, Not Green",
               "rgb-no-gray-no-green")

  mask_remove_small = filter_remove_small_objects(mask_gray_green, min_size=500, output_type="bool")
  rgb_remove_small = mask_rgb(rgb, mask_remove_small)
  save_display(save, display, info, rgb_remove_small, slide_num, 8,
               "Not Gray, Not Green,\nRemove Small Objects",
               "rgb-not-green-not-gray-remove-small")

  img = rgb_remove_small
  return img


def apply_filters_to_image(slide_num, save=True, display=False):
  """
  Apply a set of filters to an image and optionally save and/or display filtered images.
  Args:
    slide_num: The slide number.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
  Returns:
    Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
    (used for HTML page generation).
  """
  t = Time()
  print("Processing slide #%d" % slide_num)

  info = dict()

  if save and not os.path.exists(FILTER_DIR):
    os.makedirs(FILTER_DIR)
  img_path = get_training_image_path(slide_num)
  np_orig = open_image_np(img_path)
  filtered_np_img = apply_image_filters(np_orig, slide_num, info, save=save, display=display)

  if save:
    t1 = Time()
    result_path = get_filter_image_result(slide_num)
    pil_img = np_to_pil(filtered_np_img)
    pil_img.save(result_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

    t1 = Time()
    thumbnail_path = get_filter_thumbnail_result(slide_num)
    save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

  print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

  return filtered_np_img, info


def save_display(save, display, info, np_img, slide_num, filter_num, display_text, file_text,
                 display_mask_percentage=True):
  """
  Optionally save an image and/or display the image.
  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    info: Dictionary to store filter information.
    np_img: Image as a NumPy array.
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.
    display_mask_percentage: If True, display mask percentage on displayed slide.
  """
  mask_percentage = None
  if display_mask_percentage:
    mask_percentage = mask_percent(np_img)
    display_text = display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
  if slide_num is None and filter_num is None:
    pass
  elif filter_num is None:
    display_text = "S%03d " % slide_num + display_text
  elif slide_num is None:
    display_text = "F%03d " % filter_num + display_text
  else:
    display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
  if display:
    display_img(np_img, display_text)
  if save:
    save_filtered_image(np_img, slide_num, filter_num, file_text)
  if info is not None:
    info[slide_num * 1000 + filter_num] = (slide_num, filter_num, display_text, file_text, mask_percentage)


def mask_percentage_text(mask_percentage):
  """
  Generate a formatted string representing the percentage that an image is masked.
  Args:
    mask_percentage: The mask percentage.
  Returns:
    The mask percentage formatted as a string.
  """
  return "%3.2f%%" % mask_percentage


def save_filtered_image(np_img, slide_num, filter_num, filter_text):
  """
  Save a filtered image to the file system.
  Args:
    np_img: Image as a NumPy array.
    slide_num:  The slide number.
    filter_num: The filter number.
    filter_text: Descriptive text to add to the image filename.
  """
  t = Time()
  filepath = get_filter_image_path(slide_num, filter_num, filter_text)
  pil_img = np_to_pil(np_img)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))

  t1 = Time()
  thumbnail_filepath = get_filter_thumbnail_path(slide_num, filter_num, filter_text)
  save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_filepath))


def apply_filters_to_image_list(image_num_list, save, display):
  """
  Apply filters to a list of images.
  Args:
    image_num_list: List of image numbers.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
  Returns:
    Tuple consisting of 1) a list of image numbers, and 2) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in image_num_list:
    _, info = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(info)
  return image_num_list, html_page_info


def apply_filters_to_image_range(start_ind, end_ind, save, display):
  """
  Apply filters to a range of images.
  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
  Returns:
    Tuple consisting of 1) staring index of slides converted to images, 2) ending index of slides converted to images,
    and 3) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in range(start_ind, end_ind + 1):
    _, info = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(info)
  return start_ind, end_ind, html_page_info


def singleprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to training images and optionally save and/or display the filtered images.
  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Applying filters to images\n")

  if image_num_list is not None:
    _, info = apply_filters_to_image_list(image_num_list, save, display)
  else:
    num_training_slides = get_num_training_slides()
    (s, e, info) = apply_filters_to_image_range(1, num_training_slides, save, display)

  print("Time to apply filters to all images: %s\n" % str(t.elapsed()))

  if html:
    generate_filter_html_result(info)


def multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to all training images using multiple processes (one process per core).
  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen (multiprocessed display not recommended).
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Applying filters to images (multiprocess)\n")

  if save and not os.path.exists(FILTER_DIR):
    os.makedirs(FILTER_DIR)

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
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
      tasks.append((sublist, save, display))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, save, display))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(apply_filters_to_image_list, t))
    else:
      results.append(pool.apply_async(apply_filters_to_image_range, t))

  html_page_info = dict()
  for result in results:
    if image_num_list is not None:
      (image_nums, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      print("Done filtering slides: %s" % image_nums)
    else:
      (start_ind, end_ind, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      if (start_ind == end_ind):
        print("Done filtering slide %d" % start_ind)
      else:
        print("Done filtering slides %d through %d" % (start_ind, end_ind))

  if html:
    generate_filter_html_result(html_page_info)

  print("Time to apply filters to all images (multiprocess): %s\n" % str(timer.elapsed()))

# Tile.py よりmerge

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
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).
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


def generate_tile_summaries(tile_sum, np_img, display=True, save_summary=True):
  """
  Generate summary images/thumbnails showing a 'heatmap' representation of the tissue segmentation of all tiles.
  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
  """
  z = 350  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = get_training_image_path(slide_num)
  np_orig = open_image_np(original_img_path)
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

  if display:
    summary.show()
    summary_orig.show()

  if save_summary:
    save_tile_summary_image(summary, slide_num)
    save_tile_summary_on_original_image(summary_orig, slide_num)


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
  return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
         "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
         "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
         "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
           tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
         "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
         " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
           tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
           tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
           TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
           tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
         " %5d (%5.2f%%) tiles =0%% tissue\n" % (tile_summary.none, tile_summary.none / tile_summary.count * 100) + \
         "Export Tiles: threshold %d%%, compression: 1/%dx\n" % (EXPORT_TILE_THRESH, EXPORT_TILE_SCALE) + \
         "%5d (%5.2f%%) tiles exported" % (tile_summary.export, tile_summary.export / tile_summary.count * 100)


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


def save_tile_summary_image(pil_img, slide_num):
  """
  Save a tile summary image and thumbnail to the file system.
  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = get_tile_summary_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = get_tile_summary_thumbnail_path(slide_num)
  save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Thumb", str(t.elapsed()), thumbnail_filepath))


def save_tile_summary_on_original_image(pil_img, slide_num):
  """
  Save a tile summary on original image and thumbnail to the file system.
  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = get_tile_summary_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = get_tile_summary_on_original_thumbnail_path(slide_num)
  save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
  print(
    "%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig T", str(t.elapsed()), thumbnail_filepath))


def summary_and_tiles(slide_num, display=True, save_summary=False, save_data=True, save_top_tiles=True):
  """
  Generate tile summary and top tiles for slide.
  Args:
    slide_num: The slide number.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
  """
  img_path = get_filter_image_result(slide_num)
  np_img = open_image_np(img_path)

  tile_sum = score_tiles(slide_num, np_img)
  if save_data:
    save_tile_data(tile_sum)
    generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  if save_top_tiles:
    for tile in tile_sum.top_tiles():
      tile.save_tile()
  return tile_sum


def save_tile_data(tile_summary):
  """
  Save tile data to csv file.

  Args
    tile_summary: TimeSummary object.
  """

  time = Time()
  csv = summary_title(tile_summary) + "\n" + summary_stats(tile_summary) + "\n\n\n"

  csv += "Tile Num,Row,Column,Tissue %,Col Start,Row Start,Col End,Row End," + \
         "Original Col Start,Original Row Start,Original Col End,Original Row End,Score\n"

  for t in tile_summary.tiles:
    line = "%d,%d,%d,%4.2f,%d,%d,%d,%d,%d,%d,%d,%d,%0.4f\n" % (
      t.tile_num, t.r, t.c, t.tissue_percentage, t.c_s, t.r_s, t.c_e, t.r_e, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.score)
    csv += line

  data_path = get_tile_data_path(tile_summary.slide_num)
  csv_file = open(data_path, "w")
  csv_file.write(csv)
  csv_file.close()

  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Data", str(time.elapsed()), data_path))


def normalizeStaining(img,Io=240, alpha=1, beta=0.15):
  """
  Normalize staining appearence of H&E stained images
  Args:
   I: RGB input image
   Io: (optional) transmitted light intensity
  Return:
     Inorm: normalized image
     H: hematoxylin image
     E: eosin image
  """
  img = np.array(img)
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


def tile_to_pil_tile(tile):
  """
  Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.
  Args:
    tile: Tile object.
  Return:
    Tile as a PIL image.
  """
  t = tile
  slide_filepath = get_training_slide_path(t.slide_num)
  s = open_slide(slide_filepath)

  x, y = t.o_c_s, t.o_r_s
  w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
  tile_w = math.floor(w / EXPORT_TILE_SCALE)
  tile_h = math.floor(h / EXPORT_TILE_SCALE)
  tile_region = s.read_region((x, y), 0, (w, h))
  # RGBA to RGB
  tile_img = tile_region.convert("RGB")
  pil_img = tile_img.resize((tile_w, tile_h), PIL.Image.BILINEAR)
  pil_normalized_img = normalizeStaining(pil_img, Io=normalize_Io, alpha=normalize_alpha, beta=normalize_beta)
  return pil_normalized_img


def save_display_tile(tile, save=True, display=False):
  """
  Save and/or display a tile image.
  Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
  """
  tile_pil_img = tile_to_pil_tile(tile)

  if save:
    t = Time()
    img_path = get_tile_image_path(tile)
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    tile_pil_img.save(img_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile", str(t.elapsed()), img_path))

  if display:
    tile_pil_img.show()


def score_tiles(slide_num, np_img=None, dimensions=None, small_tile_in_tile=False):
  """
  Score all tiles for a slide and return the results in a TileSummary object.
  Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
    dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
      tile retrieval.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.
  Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
  """
  if dimensions is None:
    img_path = get_filter_image_result(slide_num)
    o_w, o_h, w, h = parse_dimensions_from_image_filename(img_path)
  else:
    o_w, o_h, w, h = dimensions

  if np_img is None:
    np_img = open_image_np(img_path)

  row_tile_size = round(ROW_TILE_SIZE / SCALE_FACTOR)  # use round?
  col_tile_size = round(COL_TILE_SIZE / SCALE_FACTOR)  # use round?

  num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

  tile_sum = TileSummary(slide_num=slide_num,
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

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0
  export = 0
  noexport = 0
  tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)

  for t in tile_indices:
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
    score = score_tile(np_tile, t_p, slide_num, r, c)

    np_scaled_tile = np_tile if small_tile_in_tile else None
    tile = Tile(tile_sum, slide_num, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                  o_c_e, t_p, score)
    tile_sum.tiles.append(tile)

  tile_sum.count = count
  tile_sum.high = high
  tile_sum.medium = medium
  tile_sum.low = low
  tile_sum.none = none
  tile_sum.export = export
  tile_sum.noexport = noexport

  tiles_by_score = tile_sum.tiles_by_score()
  rank = 0
  for t in tiles_by_score:
    rank += 1
    t.rank = rank

  return tile_sum


def score_tile(np_tile, tissue_percent, slide_num, row, col):
  """
  Score tile based on tissue percentage.
  Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.
    row: Tile row.
    col: Tile column.
  Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
  """
  score = tissue_percent / 100
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


def image_list_to_tiles(image_num_list, display=False, save_summary=True, save_data=True, save_top_tiles=True):
  """
  Generate tile summaries and tiles for a list of images.
  Args:
    image_num_list: List of image numbers.
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
  """
  tile_summaries_dict = dict()
  for slide_num in image_num_list:
    tile_summary = summary_and_tiles(slide_num, display, save_summary, save_data, save_top_tiles)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def image_range_to_tiles(start_ind, end_ind, display=False, save_summary=True, save_data=True, save_top_tiles=True):
  """
  Generate tile summaries and tiles for a range of images.
  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
  """
  image_num_list = list()
  tile_summaries_dict = dict()
  for slide_num in range(start_ind, end_ind + 1):
    tile_summary = summary_and_tiles(slide_num, display, save_summary, save_data, save_top_tiles)
    image_num_list.append(slide_num)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def singleprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True, save_top_tiles=True,
                                           html=True, image_num_list=None):
  """
  Generate tile summaries and tiles for training images using a single process.
  Args:
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
    html: If True, generate HTML page to display tiled images
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Generating tile summaries\n")

  if image_num_list is not None:
    image_num_list, tile_summaries_dict = image_list_to_tiles(image_num_list, display, save_summary, save_data,
                                                              save_top_tiles)
  else:
    num_training_slides = get_num_training_slides()
    image_num_list, tile_summaries_dict = image_range_to_tiles(1, num_training_slides, display, save_summary, save_data,
                                                               save_top_tiles)

  print("Time to generate tile summaries: %s\n" % str(t.elapsed()))

  if html:
    generate_tiled_html_result(image_num_list, tile_summaries_dict, save_data)


def multiprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True, save_top_tiles=True,
                                          html=True, image_num_list=None):
  """
  Generate tile summaries and tiles for all training images using multiple processes (one process per core).
  Args:
    display: If True, display images to screen (multiprocessed display not recommended).
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
    html: If True, generate HTML page to display tiled images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Generating tile summaries (multiprocess)\n")

  if save_summary and not os.path.exists(TILE_SUMMARY_DIR):
    os.makedirs(TILE_SUMMARY_DIR)

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
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
      tasks.append((sublist, display, save_summary, save_data, save_top_tiles))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, display, save_summary, save_data, save_top_tiles))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(image_list_to_tiles, t))
    else:
      results.append(pool.apply_async(image_range_to_tiles, t))

  slide_nums = list()
  tile_summaries_dict = dict()
  for result in results:
    image_nums, tile_summaries = result.get()
    slide_nums.extend(image_nums)
    tile_summaries_dict.update(tile_summaries)
    print("Done tiling slides: %s" % image_nums)

  if html:
    generate_tiled_html_result(slide_nums, tile_summaries_dict, save_data)

  print("Time to generate tile previews (multiprocess): %s\n" % str(timer.elapsed()))


def display_image(np_rgb, text=None, scale_up=False):
  """
  Display an image with optional text above image.
  Args:
    np_rgb: RGB image tile as a NumPy array
    text: Optional text to display above image
    scale_up: If True, scale up image to display by slide.SCALE_FACTOR
  """
  if scale_up:
    np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i

  pil_img = np_to_pil(np_rgb)
  pil_img.show()


def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  """
  Obtain a PIL image representation of text.
  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.
  Returns:
    PIL image representing the text.
  """

  font = ImageFont.truetype(font_path, font_size)
  x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
  image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
  draw = ImageDraw.Draw(image)
  draw.text((w_border, h_border), text, text_color, font=font)
  return image


def np_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
            font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  """
  Obtain a NumPy array image representation of text.
  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.
  Returns:
    NumPy array representing the text.
  """
  pil_img = pil_text(text, w_border, h_border, font_path, font_size,
                     text_color, background)
  np_img = pil_to_np_rgb(pil_img)
  return np_img


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

  def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
    self.slide_num = slide_num
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

  def mask_percentage(self):
    """
    Obtain the percentage of the slide that is masked.
    Returns:
       The amount of the slide that is masked as a percentage.
    """
    return 100 - self.tissue_percentage

  def num_tiles(self):
    """
    Retrieve the total number of tiles.
    Returns:
      The total number of tiles (number of rows * number of columns).
    """
    return self.num_row_tiles * self.num_col_tiles


  def tiles_by_score(self):
    """
    Retrieve the tiles ranked by score.
    Returns:
       List of the tiles ranked by score.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
    return sorted_list


  def top_tiles(self):
    """
    Retrieve the top-scoring tiles.
    Returns:
       List of the top-scoring tiles.
    """
    sorted_tiles = self.tiles_by_score()
    top_tiles = filter(lambda t: t.score > (EXPORT_TILE_THRESH/100), sorted_tiles)
    return top_tiles


  def get_tile(self, row, col):
    """
    Retrieve tile by row and column.
    Args:
      row: The row
      col: The column
    Returns:
      Corresponding Tile object.
    """
    tile_index = (row - 1) * self.num_col_tiles + (col - 1)
    tile = self.tiles[tile_index]
    return tile

  def display_summaries(self):
    """
    Display summary images.
    """
    summary_and_tiles(self.slide_num, display=True, save_summary=False, save_data=False, save_top_tiles=False)


class Tile:
  """
  Class for information about a tile.
  """
  def __init__(self, tile_summary, slide_num, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, score):
    self.tile_summary = tile_summary
    self.slide_num = slide_num
    self.np_scaled_tile = np_scaled_tile
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
    self.tissue_percentage = t_p
    self.score = score

  def __str__(self):
    return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

  def __repr__(self):
    return "\n" + self.__str__()

  def mask_percentage(self):
    return 100 - self.tissue_percentage

  def tissue_quantity(self):
    return tissue_quantity(self.tissue_percentage)

  def tissue_export(self):
    return tissue_export(self.tissue_percentage)

  def get_pil_tile(self):
    return tile_to_pil_tile(self)

  def save_tile(self):
    save_display_tile(self, save=True, display=False)

  def display_tile(self):
    save_display_tile(self, save=False, display=True)

  def get_np_scaled_tile(self):
    return self.np_scaled_tile

  def get_pil_scaled_tile(self):
    return np_to_pil(self.np_scaled_tile)


class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3


class TissueExport(Enum):
  EXPORT = 0
  NOEXPORT = 1


## HTML export tools
  """
  filter, tile 処理情報 の html出力方法
  """
def html_header(page_title):
    """
    Generate an HTML header for previewing images.
    Returns:
      HTML header for viewing images.
    """
    html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" " + \
           "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n" + \
           "<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"en\" xml:lang=\"en\">\n" + \
           "  <head>\n" + \
           "    <title>%s</title>\n" % page_title + \
           "    <style type=\"text/css\">\n" + \
           "     img { border: 2px solid black; }\n" + \
           "     td { border: 2px solid black; }\n" + \
           "    </style>\n" + \
           "  </head>\n" + \
           "  <body>\n"
    return html


def html_footer():
  """
  Generate an HTML footer for previewing images.
  Returns:
    HTML footer for viewing images.
  """
  html = "</body>\n" + \
         "</html>\n"
  return html


# filter information
def image_cell(slide_num, filter_num, display_text, file_text):
    """
    Generate HTML for viewing a processed image.
    Args:
      slide_num: The slide number.
      filter_num: The filter number.
      display_text: Filter display name.
      file_text: Filter name for file.
    Returns:
      HTML for a table cell for viewing a filtered image.
    """
    filt_img = get_filter_image_path(slide_num, filter_num, file_text)
    filt_thumb = get_filter_thumbnail_path(slide_num, filter_num, file_text)
    img_name = get_filter_image_filename(slide_num, filter_num, file_text)
    return "      <td>\n" + \
           "        <a target=\"_blank\" href=\"%s\">%s<br/>\n" % (filt_img, display_text) + \
           "          <img src=\"%s\" />\n" % (filt_thumb) + \
           "        </a>\n" + \
           "      </td>\n"

def generate_filter_html_result(html_page_info):
    """
    Generate HTML to view the filtered images. If FILTER_PAGINATE is True, the results will be paginated.
    Args:
      html_page_info: Dictionary of image information.
    """
    if not FILTER_PAGINATE:
      html = ""
      html += html_header("Filtered Images")
      html += "  <table>\n"
      row = 0
      for key in sorted(html_page_info):
        value = html_page_info[key]
        current_row = value[0]
        if current_row > row:
          html += "    <tr>\n"
          row = current_row
        html += image_cell(value[0], value[1], value[2], value[3])
        next_key = key + 1
        if next_key not in html_page_info:
          html += "    </tr>\n"
      html += "  </table>\n"
      html += html_footer()
      text_file = open(os.path.join(FILTER_HTML_DIR, "filters.html"), "w")
      text_file.write(html)
      text_file.close()
    else:
      slide_nums = set()
      for key in html_page_info:
        slide_num = math.floor(key / 1000)
        slide_nums.add(slide_num)
      slide_nums = sorted(list(slide_nums))
      total_len = len(slide_nums)
      page_size = FILTER_PAGINATION_SIZE
      num_pages = math.ceil(total_len / page_size)

      for page_num in range(1, num_pages + 1):
        start_index = (page_num - 1) * page_size
        end_index = (page_num * page_size) if (page_num < num_pages) else total_len
        page_slide_nums = slide_nums[start_index:end_index]

        html = ""
        html += html_header("Filtered Images (Page %d)" % page_num)

        html += "  <div style=\"font-size: 20px\">"
        if page_num > 1:
          if page_num == 2:
            html += "<a href=\"filters.html\">&lt;</a> "
          else:
            html += "<a href=\"filters-%d.html\">&lt;</a> " % (page_num - 1)
        html += "Page %d" % page_num
        if page_num < num_pages:
          html += " <a href=\"filters-%d.html\">&gt;</a> " % (page_num + 1)
        html += "</div>\n"

        html += "  <table>\n"
        for slide_num in page_slide_nums:
          html += "  <tr>\n"
          filter_num = 1

          lookup_key = slide_num * 1000 + filter_num
          while lookup_key in html_page_info:
            value = html_page_info[lookup_key]
            html += image_cell(value[0], value[1], value[2], value[3])
            lookup_key += 1
          html += "  </tr>\n"

        html += "  </table>\n"

        html += html_footer()
        if page_num == 1:
          text_file = open(os.path.join(FILTER_HTML_DIR, "filters.html"), "w")
        else:
          text_file = open(os.path.join(FILTER_HTML_DIR, "filters-%d.html" % page_num), "w")
        text_file.write(html)
        text_file.close()


# Tile information
def image_row(slide_num, tile_summary, data_link):
    """
    Generate HTML for viewing a tiled image.
    Args:
      slide_num: The slide number.
      tile_summary: TileSummary object.
      data_link: If True, add link to tile data csv file.
    Returns:
      HTML table row for viewing a tiled image.
    """
    orig_img = get_training_image_path(slide_num)
    orig_thumb = get_training_thumbnail_path(slide_num)
    filt_img = get_filter_image_result(slide_num)
    filt_thumb = get_filter_thumbnail_result(slide_num)
    sum_img = get_tile_summary_image_path(slide_num)
    sum_thumb = get_tile_summary_thumbnail_path(slide_num)
    osum_img = get_tile_summary_on_original_image_path(slide_num)
    osum_thumb = get_tile_summary_on_original_thumbnail_path(slide_num)

    html = "    <tr>\n" + \
           "      <td style=\"vertical-align: top\">\n" + \
           "        <a target=\"_blank\" href=\"%s\">S%03d Original<br/>\n" % (orig_img, slide_num) + \
           "          <img src=\"%s\" />\n" % (orig_thumb) + \
           "        </a>\n" + \
           "      </td>\n" + \
           "      <td style=\"vertical-align: top\">\n" + \
           "        <a target=\"_blank\" href=\"%s\">S%03d Filtered<br/>\n" % (filt_img, slide_num) + \
           "          <img src=\"%s\" />\n" % (filt_thumb) + \
           "        </a>\n" + \
           "      </td>\n"

    html += "      <td style=\"vertical-align: top\">\n" + \
            "        <a target=\"_blank\" href=\"%s\">S%03d Tiles<br/>\n" % (sum_img, slide_num) + \
            "          <img src=\"%s\" />\n" % (sum_thumb) + \
            "        </a>\n" + \
            "      </td>\n"

    html += "      <td style=\"vertical-align: top\">\n" + \
            "        <a target=\"_blank\" href=\"%s\">S%03d Tiles<br/>\n" % (osum_img, slide_num) + \
            "          <img src=\"%s\" />\n" % (osum_thumb) + \
            "        </a>\n" + \
            "      </td>\n"

    html += "      <td style=\"vertical-align: top\">\n"
    if data_link:
      html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary\n" % slide_num + \
              "        (<a target=\"_blank\" href=\"%s\">Data</a>)</div>\n" % get_tile_data_path(slide_num)
    else:
      html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary</div>\n" % slide_num

    html += "        <div style=\"font-size: smaller; white-space: nowrap;\">\n" + \
            "          %s\n" % summary_stats(tile_summary).replace("\n", "<br/>\n          ") + \
            "        </div>\n" + \
            "      </td>\n"
    html += "    </tr>\n"
    return html

def generate_tiled_html_result(slide_nums, tile_summaries_dict, data_link):
    """
    Generate HTML to view the tiled images.
    Args:
      slide_nums: List of slide numbers.
      tile_summaries_dict: Dictionary of TileSummary objects keyed by slide number.
      data_link: If True, add link to tile data csv file.
    """
    slide_nums = sorted(slide_nums)
    if not TILE_SUMMARY_PAGINATE:
      html = ""
      html += html_header("Tiles Summary")

      html += "  <table>\n"
      for slide_num in slide_nums:
        html += image_row(slide_num, data_link)
      html += "  </table>\n"

      html += html_footer()
      text_file = open(os.path.join(TILE_SUMMARY_HTML_DIR, "tiles.html"), "w")
      text_file.write(html)
      text_file.close()
    else:
      total_len = len(slide_nums)
      page_size = TILE_SUMMARY_PAGINATION_SIZE
      num_pages = math.ceil(total_len / page_size)
      for page_num in range(1, num_pages + 1):
        start_index = (page_num - 1) * page_size
        end_index = (page_num * page_size) if (page_num < num_pages) else total_len
        page_slide_nums = slide_nums[start_index:end_index]

        html = ""
        html += html_header("Tiles Summary (Page %d)" % page_num)

        html += "  <div style=\"font-size: 20px\">"
        if page_num > 1:
          if page_num == 2:
            html += "<a href=\"tiles.html\">&lt;</a> "
          else:
            html += "<a href=\"tiles-%d.html\">&lt;</a> " % (page_num - 1)
        html += "Page %d" % page_num
        if page_num < num_pages:
          html += " <a href=\"tiles-%d.html\">&gt;</a> " % (page_num + 1)
        html += "</div>\n"

        html += "  <table>\n"
        for slide_num in page_slide_nums:
          tile_summary = tile_summaries_dict[slide_num]
          html += image_row(slide_num, tile_summary, data_link)
        html += "  </table>\n"

        html += html_footer()
        if page_num == 1:
          text_file = open(os.path.join(TILE_SUMMARY_HTML_DIR, "tiles.html"), "w")
        else:
          text_file = open(os.path.join(TILE_SUMMARY_HTML_DIR, "tiles-%d.html" % page_num), "w")
        text_file.write(html)
        text_file.close()

## 実行コード
multiprocess_training_slides_to_images()
multiprocess_apply_filters_to_images()
multiprocess_filtered_images_to_tiles()