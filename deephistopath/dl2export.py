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

import os, sys, re, glob, math
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
from openslide import OpenSlideError
from PIL import Image, ImageDraw, ImageFont
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
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg19 import VGG19
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.xception import Xception
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications import NASNetLarge

## 画像処理設定項目：
# FILTER処理を行う画像の縮小倍率 (例 2 → 縦1/2, 横 1/2で書き出し)
SCALE_FACTOR = 32
# 分割する際の1枚あたりのタイルサイズ
ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
# Tileサマリ画像上でのラベル有無設定
DISPLAY_TILE_SUMMARY_LABELS = True
# Label names
classes = ["IR", "MT", "PG", "SP", "ST"]
# 学習モデル名を記載
MODEL_NAME = "model_ResNet50"

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

TILE_ANNOTATION_SUMMARY_DIR = os.path.join(BASE_DIR, "tile_summary_annotation_" + DEST_TRAIN_EXT)
TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, "tile_summary_annotation_on_original_" + DEST_TRAIN_EXT)
TILE_ANNOTATION_SUMMARY_SUFFIX = "tile_summary_annotation"
TILE_ANNOTATION_SUMMARY_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_annotation_thumbnail_" + THUMBNAIL_EXT)
TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_annotation_on_original_thumbnail_" + THUMBNAIL_EXT)
TILE_ANNOTATION_SUMMARY_PAGINATION_SIZE = 50
TILE_ANNOTATION_SUMMARY_PAGINATE = True
TILE_ANNOTATION_SUMMARY_HTML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_ANNOTATION_DIR = os.path.join(BASE_DIR, "tile_annotation")
TILE_ANNOTATION_SUFFIX = "tile_annotation"

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/model/"

## Exportファイルの色設定
IR_COLOR = (72, 61, 139)
MT_COLOR = (220,20,60)
SP_COLOR = (218,165,32)
PG_COLOR = (255,0,255)
ST_COLOR = (232,190,193)
ND_COLOR = (0, 0, 0)

TILE_LABEL_TEXT_SIZE = 10
TILE_BORDER_SIZE = 2

FONT_PATH = "/Library/Fonts/Arial Bold.ttf"
SUMMARY_TITLE_FONT_PATH = "/Library/Fonts/Courier New Bold.ttf"
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
ANNOTATION_SUMMARY_TILE_TEXT_COLOR = (0, 0, 0)
ANNOTATION_SUMMARY_TILE_TEXT_COLOR_ORI = (255, 255, 255)

TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4


def open_image_np(filename):
  """
  Open an image (*.jpg, *.png, etc) as an RGB NumPy array.
  Args:
    filename: Name of the image file.
  returns:
    A NumPy representing an RGB image.
  """
  pil_img = Image.open(filename)
  np_img = pil_to_np_rgb(pil_img)
  return np_img


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


def get_tile_annotation_summary_image_path(slide_number):
  """
  Convert slide number to a path to a annotation summary image file.
  Example:
    5 -> ../images/tile_summary_png/HGSOC_005-tile_summary_annotation.png
  Args:
    slide_number: The slide number.
  Returns:
    Path to the annotation summary image file.
  """
  if not os.path.exists(TILE_ANNOTATION_SUMMARY_DIR):
    os.makedirs(TILE_ANNOTATION_SUMMARY_DIR)
  img_path = os.path.join(TILE_ANNOTATION_SUMMARY_DIR, get_tile_annotation_summary_image_filename(slide_number))
  return img_path


def get_tile_annotation_summary_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a annotation summary thumbnail file.
  Example:
    5 -> ../images/tile_summary_thumbnail_jpg/HGSOC_005-tile_summary_annotation.jpg
  Args:
    slide_number: The slide number.
  Returns:
    Path to the annotation summary thumbnail file.
  """
  if not os.path.exists(TILE_ANNOTATION_SUMMARY_THUMBNAIL_DIR):
    os.makedirs(TILE_ANNOTATION_SUMMARY_THUMBNAIL_DIR)
  img_path = os.path.join(TILE_ANNOTATION_SUMMARY_THUMBNAIL_DIR, get_tile_annotation_summary_image_filename(slide_number, thumbnail=True))
  return img_path


def get_tile_annotation_summary_on_original_image_path(slide_number):
  """
  Convert slide number to a path to a annotation summary on original image file.
  Example:
    5 -> ../images/tile_summary_on_original_png/HGSOC_005-tile_summary_annotation.png
  Args:
    slide_number: The slide number.
  Returns:
    Path to the annotation summary on original image file.
  """
  if not os.path.exists(TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_DIR):
    os.makedirs(TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_DIR)
  img_path = os.path.join(TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_DIR, get_tile_annotation_summary_image_filename(slide_number))
  return img_path


def get_tile_annotation_summary_on_original_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a annotation summary on original thumbnail file.
  Example:
    5 -> ../images/tile_summary_on_original_thumbnail_jpg/HGSOC_005-tile_summary_annotation.jpg
  Args:
    slide_number: The slide number.
  Returns:
    Path to the annotation summary on original thumbnail file.
  """
  if not os.path.exists(TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR):
    os.makedirs(TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR)
  img_path = os.path.join(TILE_ANNOTATION_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR,
                          get_tile_annotation_summary_image_filename(slide_number, thumbnail=True))
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



def get_tile_annotation_summary_image_filename(slide_number, thumbnail=False):
  """
  Convert slide number to a annotation summary image file name.
  Example:
    5, False -> HGSOC_005-tile_summary.png
    5, True -> HGSOC_005-tile_summary.jpg
  Args:
    slide_number: The slide number.
    thumbnail: If True, produce thumbnail filename.
  Returns:
    The annotation summary image file name.
  """
  if thumbnail:
    ext = THUMBNAIL_EXT
  else:
    ext = DEST_TRAIN_EXT
  padded_sl_num = str(slide_number).zfill(3)

  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_ANNOTATION_SUMMARY_SUFFIX + "." + ext
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


def get_tile_annotation_data_filename(slide_number):
  """
  Convert slide number to a annotation data file name.
  Example:
    5 -> HGSOC_005-32x-49920x108288-1560x3384-tile_annotation.csv
  Args:
    slide_number: The slide number.
  Returns:
    The tile data file name.
  """
  padded_sl_num = str(slide_number).zfill(3)

  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  data_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_ANNOTATION_SUFFIX + ".csv"

  return data_filename


def get_tile_annotation_path(slide_number):
  """
  Convert slide number to a path to a annotation data file.
  Example:
    5 -> ../images/tile_data/HGSOC_005-32x-49920x108288-1560x3384-tile_annotation.csv
  Args:
    slide_number: The slide number.
  Returns:
    Path to the tile data file.
  """
  if not os.path.exists(TILE_ANNOTATION_DIR):
    os.makedirs(TILE_ANNOTATION_DIR)
  file_path = os.path.join(TILE_ANNOTATION_DIR, get_tile_annotation_data_filename(slide_number))
  return file_path


def get_num_training_slides():
  """
  Obtain the total number of WSI training slide images.
  Returns:
    The total number of WSI training slide images.
  """
  num_training_slides = len(glob.glob1(SRC_TRAIN_DIR, "*." + SRC_TRAIN_EXT))
  return num_training_slides


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


# def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
#                 background=(255, 255, 255), border=(0, 0, 0), bg=False):
#   """
#   Convert a NumPy array to a PIL image, add text to the image, and display the image.
#   Args:
#     np_img: Image as a NumPy array.
#     text: The text to add to the image.
#     font_path: The path to the font to use.
#     size: The font size
#     color: The font color
#     background: The background color
#     border: The border color
#     bg: If True, add rectangle background behind text
#   """
#   result = np_to_pil(np_img)
#   # if gray, convert to RGB for display
#   if result.mode == 'L':
#     result = result.convert('RGB')
#   draw = ImageDraw.Draw(result)
#   if text is not None:
#     font = ImageFont.truetype(font_path, size)
#     if bg:
#       (x, y) = draw.textsize(text, font)
#       draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
#     draw.text((2, 0), text, color, font=font)
#   result.show()


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
  img = pil_img.resize(max_size, Image.BILINEAR)
  if display_path:
    print("Saving thumbnail to: " + path)
  dir = os.path.dirname(path)
  if dir != '' and not os.path.exists(dir):
    os.makedirs(dir)
  img.save(path)


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


def generate_tile_annotation_summaries(tile_sum_annotation, np_img, display=True, save_summary=True):
  """
  Generate summary images/thumbnails showing a 'heatmap' representation of the tissue segmentation of all tiles.
  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
  """
  z = 350  # height of area at top of summary slide
  slide_num = tile_sum_annotation.slide_num
  rows = tile_sum_annotation.scaled_h
  cols = tile_sum_annotation.scaled_w
  row_tile_size = tile_sum_annotation.scaled_tile_h
  col_tile_size = tile_sum_annotation.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = get_training_image_path(slide_num)
  np_orig = open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  for t in tile_sum_annotation.tilesannotation:
    border_color = tile_annotation_border_color(t.tissue_label)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

  summary_txt = summary_title(tile_sum_annotation) + "\n" + summary_annotation_stats(tile_sum_annotation)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  if DISPLAY_TILE_SUMMARY_LABELS:
    count = 0
    for t in tile_sum_annotation.tilesannotation:
      count += 1
      label = "%s" % (t.tissue_label)
      font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
      # drop shadow behind text
      draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)
      draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)

      draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, ANNOTATION_SUMMARY_TILE_TEXT_COLOR, font=font)
      draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label,ANNOTATION_SUMMARY_TILE_TEXT_COLOR_ORI, font=font)

  if display:
    summary.show()
    summary_orig.show()

  if save_summary:
    save_tile_summary_annotation_image(summary, slide_num)
    save_tile_summary_annotation_on_original_image(summary_orig, slide_num)


def tile_annotation_border_color(tissue_label):
  """
  Obtain the corresponding tile border color for a particular tile tissue label.
  Args:
    tissue_label: The tile tissue label
  Returns:
    The tile border color corresponding to the tile tissue label.
  """
  if tissue_label == "IR":
    border_color = IR_COLOR
  elif tissue_label == "MT":
    border_color = MT_COLOR
  elif tissue_label == "PG":
    border_color = PG_COLOR
  elif tissue_label == "SP":
    border_color = SP_COLOR
  elif tissue_label == "ST":
    border_color = ST_COLOR
  else:
    border_color = ND_COLOR
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


def summary_annotation_stats(tile_summary_annotation):
  """
  Obtain various stats about the slide tiles.
  Args:
    tile_summary: TileSummary object.
  Returns:
     Various stats about the slide tiles as a string.
  """
  return "Original Dimensions: %dx%d\n" % (tile_summary_annotation.orig_w, tile_summary_annotation.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary_annotation.orig_tile_w, tile_summary_annotation.orig_tile_h) + \
         "Scale Factor: 1/%dx\n" % tile_summary_annotation.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary_annotation.scaled_w, tile_summary_annotation.scaled_h) + \
         "Tiles: %d labeled\n" % (tile_summary_annotation.count-tile_summary_annotation.none) + \
         " Tumor = %5d tiles (%5.2f%%/labeled)\n" % (
           tile_summary_annotation.tumor, tile_summary_annotation.tumor/(tile_summary_annotation.count-tile_summary_annotation.none) * 100) + \
         "  - Label: IR = %5d (%5.2f%%) tiles\n" % (
           tile_summary_annotation.IR, tile_summary_annotation.IR / tile_summary_annotation.tumor * 100) + \
         "  - Label: MT = %5d (%5.2f%%) tiles\n" % (
           tile_summary_annotation.MT, tile_summary_annotation.MT / tile_summary_annotation.tumor * 100) + \
         "  - Label: PG = %5d (%5.2f%%) tiles\n" % (
           tile_summary_annotation.PG, tile_summary_annotation.PG / tile_summary_annotation.tumor * 100) + \
         "  - Label: SP = %5d (%5.2f%%) tiles\n" % (
           tile_summary_annotation.SP, tile_summary_annotation.SP / tile_summary_annotation.tumor * 100) + \
         " Stroma = %5d tiles (%5.2f%%/labeled)\n" % (
           tile_summary_annotation.ST, tile_summary_annotation.ST / (tile_summary_annotation.count-tile_summary_annotation.none) * 100) + \
         "Estimated Tissue Subtype: %s" % (tile_summary_annotation.diagnosis)

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


def save_tile_summary_annotation_image(pil_img, slide_num):
  """
  Save a tile summary image and thumbnail to the file system.
  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = get_tile_annotation_summary_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Annotation Sum", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = get_tile_annotation_summary_thumbnail_path(slide_num)
  save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Annotation Sum Thumb", str(t.elapsed()), thumbnail_filepath))


def save_tile_summary_annotation_on_original_image(pil_img, slide_num):
  """
  Save a tile summary on original image and thumbnail to the file system.
  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = get_tile_annotation_summary_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Annotation Sum Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = get_tile_annotation_summary_on_original_thumbnail_path(slide_num)
  save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
  print(
    "%-20s | Time: %-14s  Name: %s" % ("Save Annotation Sum Orig T", str(t.elapsed()), thumbnail_filepath))


def summary_and_tiles_annotation(slide_num, display=True, save_summary=True, save_data=True):
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
  tile_sum_annotation = get_tile_annotation_summaries(slide_num)
  if save_data:
    save_annotation_data(tile_sum_annotation)
    generate_tile_annotation_summaries(tile_sum_annotation, np_img, display=display, save_summary=save_summary)
  return tile_sum_annotation


def save_annotation_data(tile_summary):
  """
  Save tile data to csv file.

  Args
    tile_summary: TimeSummary object.
  """

  time = Time()
  csv = summary_title(tile_summary) + "\n" + summary_annotation_stats(tile_summary) + "\n"

  csv += "Tile_Num,Row,Column,Tissue_%,Col_Start,Row_Start,Col_End,Row_End," + \
         "Original_Col_Start,Original_Row_Start,Original_Col_End,Original_Row_End,Score,Tissue_Label,Probability\n"

  for t in tile_summary.tilesannotation:
    line = "%d,%d,%d,%4.2f,%d,%d,%d,%d,%d,%d,%d,%d,%0.4f,%s,%s\n" % (
      t.tile_num, t.r, t.c, t.tissue_percentage, t.c_s, t.r_s, t.c_e, t.r_e, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.score, t.tissue_label, t.tissue_prob)
    csv += line

  data_path = get_tile_annotation_path(tile_summary.slide_num)
  csv_file = open(data_path, "w")
  csv_file.write(csv)
  csv_file.close()

  print("%-20s | Time: %-14s  Name: %s" % ("Save Annotation Data", str(time.elapsed()), data_path))


def get_tile_annotation_summaries(slide_num):
  """
  Score all tiles for a slide and return the results in a TileSummary object.
  Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
  Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
  """
  data_path = get_tile_data_path(slide_num)
  o_w, o_h, w, h = parse_dimensions_from_image_filename(data_path)
  num_row_tiles, num_col_tiles = get_num_tiles(o_h, o_w, ROW_TILE_SIZE, COL_TILE_SIZE)
  row_tile_size = math.floor(ROW_TILE_SIZE / SCALE_FACTOR)
  col_tile_size = math.floor(COL_TILE_SIZE / SCALE_FACTOR)
  tile_sum_annotation = TileSummaryAnnotation(slide_num=slide_num,
                         orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=row_tile_size,
                         scaled_tile_h=col_tile_size,
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)
  count = 0
  IR = 0
  MT = 0
  PG = 0
  SP = 0
  ST = 0
  ND = 0

  # import models and annotate labels
  model = load_model(MODEL_DIR + MODEL_NAME + '.hdf5', compile=False)
  df = pd.read_csv(data_path,sep=",",header=14, index_col=0)
  df["Tile_Path"] = df["Tile_Path"].astype("str")
  tile_annotation = []
  for row in df.itertuples(name=None):
    count += 1  # tile_num
    tile_num, r, c, t_p, c_s, r_s, c_e, r_e, o_c_s, o_r_s, o_c_e, o_r_e, score, tile_path = row
    if tile_path == "nan":
      tissue_label= "-"
      tissue_prob = "-"
      ND += 1
    else:
      img_path = TILE_DIR + "/" + tile_path
      image = open_image_np(img_path).astype('float32')/255
      image = image[None, ...]
      pred = model.predict(image, batch_size =1, verbose=0)
      tissue_label = classes[np.argmax(pred[0])]
      pred_prob = np.max(pred)
      tissue_prob = '{:.4f}'.format(pred_prob)
      if tissue_label == "IR":
        IR += 1
      elif tissue_label == "MT":
        MT += 1
      elif tissue_label == "PG":
        PG += 1
      elif tissue_label == "SP":
        SP += 1
      elif tissue_label == "ST":
        ST += 1
      print("Annotate tile #"+str(tile_num)+" label: " + str(tissue_label) +" probability: " + str(tissue_prob))
    tile_annotation = TileAnnotation(tile_sum_annotation, slide_num, tile_num, r, c, t_p, c_s, r_s, c_e, r_e,
         o_c_s, o_r_s, o_c_e, o_r_e, score, tile_path, tissue_label, tissue_prob)
    tile_sum_annotation.tilesannotation.append(tile_annotation)

  summary = {'IR': IR, 'MT': MT, 'PG': PG, 'SP': SP}
  tile_sum_annotation.count = count
  tile_sum_annotation.IR = IR
  tile_sum_annotation.MT = MT
  tile_sum_annotation.PG = PG
  tile_sum_annotation.SP = SP
  tile_sum_annotation.ST = ST
  tile_sum_annotation.none = ND
  tile_sum_annotation.tumor = IR+MT+PG+SP
  if MT/(IR+MT+PG+SP) >= 0.1:
    tile_sum_annotation.diagnosis = 'MT'
  else:
    tile_sum_annotation.diagnosis = max(summary.items(), key=lambda x: x[1])[0]

  return tile_sum_annotation


def annotation_list_to_tiles(image_num_list, display=False, save_summary=True, save_data=True):
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
    tile_summary = summary_and_tiles_annotation(slide_num, display, save_summary, save_data)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def annotation_range_to_tiles(start_ind, end_ind, display=False, save_summary=True, save_data=True):
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
    tile_summary = summary_and_tiles_annotation(slide_num, display, save_summary, save_data)
    image_num_list.append(slide_num)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def singleprocess_annotation_to_tiles(start_ind, display=False, save_summary=True, save_data=True,
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
  print("Generating tile annotation summaries\n")

  if image_num_list is not None:
    image_num_list, tile_summaries_dict = annotation_list_to_tiles(image_num_list, display, save_summary, save_data)
  else:
    num_training_slides = get_num_training_slides()
    image_num_list, tile_summaries_dict = annotation_range_to_tiles(start_ind, start_ind+num_training_slides-1, display, save_summary, save_data)

  print("Time to generate tile annotation summaries: %s\n" % str(t.elapsed()))

  if html:
    generate_tiled_annotation_html_result(image_num_list, tile_summaries_dict, save_data)


def multiprocess_annotation_to_tiles(display=False, save_summary=True, save_data=True,
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

  if save_summary and not os.path.exists(TILE_ANNOTATION_SUMMARY_DIR):
    os.makedirs(TILE_ANNOTATION_SUMMARY_DIR)

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
      tasks.append((sublist, display, save_summary, save_data))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, display, save_summary, save_data))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(annotation_list_to_tiles, t))
    else:
      results.append(pool.apply_async(annotation_range_to_tiles, t))

  slide_nums = list()
  tile_summaries_dict = dict()
  for result in results:
    image_nums, tile_summaries = result.get()
    slide_nums.extend(image_nums)
    tile_summaries_dict.update(tile_summaries)
    print("Done annotation slides: %s" % image_nums)

  if html:
    generate_tiled_annotation_html_result(slide_nums, tile_summaries_dict, save_data)

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
  pil_img = pil_text(text, w_border, h_border, font_path, font_size,text_color, background)
  np_img = pil_to_np_rgb(pil_img)
  return np_img


class TileSummaryAnnotation:
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
  num_row_tiles = None
  num_col_tiles = None

  IR = 0
  MT = 0
  SP = 0
  PG = 0
  ST = 0
  ND = 0

  def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h,scaled_tile_w,scaled_tile_h, num_col_tiles, num_row_tiles):
    self.slide_num = slide_num
    self.orig_w = orig_w
    self.orig_h = orig_h
    self.orig_tile_w = orig_tile_w
    self.orig_tile_h = orig_tile_h
    self.scaled_w = scaled_w
    self.scaled_h = scaled_h
    self.scaled_tile_w=scaled_tile_w
    self.scaled_tile_h=scaled_tile_h
    self.num_col_tiles = num_col_tiles
    self.num_row_tiles = num_row_tiles
    self.tilesannotation = []

  def __str__(self):
    return summary_title(self) + "\n" + summary_annotation_stats(self)

  def display_summaries(self):
    """
    Display summary images.
    """
    summary_and_tiles_annotation(self.slide_num, display=True, save_summary=False, save_data=False)

class TileAnnotation:
  """
  Class for information about a tile.
  """
  def __init__(self, tile_summary_annotation, slide_num, tile_num, r, c, t_p, c_s, r_s, c_e, r_e,
               o_c_s, o_r_s, o_c_e, o_r_e, score, tile_path, tissue_label, tissue_prob):
    self.tile_summary_annotation = tile_summary_annotation
    self.slide_num = slide_num
    self.tile_num = tile_num
    self.r = r
    self.c = c
    self.tissue_percentage = t_p
    self.c_s = c_s
    self.r_s = r_s
    self.c_e = c_e
    self.r_e = r_e
    self.o_c_s = o_c_s
    self.o_r_s = o_r_s
    self.o_c_e = o_c_e
    self.o_r_e = o_r_e
    self.score = score
    self.tile_path = tile_path
    self.tissue_label = tissue_label
    self.tissue_prob = tissue_prob

  def __str__(self):
    return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Label %s, Probability %4.2f%%]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.tissue_label, self.tissue_prob)

  def __repr__(self):
    return "\n" + self.__str__()

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


# Tile information
def image_row_annotation(slide_num, tile_summary, data_link):
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
    sum_annotation_img = get_tile_annotation_summary_image_path(slide_num)
    sum_annotation_thumb = get_tile_annotation_summary_thumbnail_path(slide_num)
    osum_annotation_img = get_tile_annotation_summary_on_original_image_path(slide_num)
    osum_annotation_thumb = get_tile_annotation_summary_on_original_thumbnail_path(slide_num)

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
            "        <a target=\"_blank\" href=\"%s\">S%03d Annotation Tiles<br/>\n" % (sum_annotation_img, slide_num) + \
            "          <img src=\"%s\" />\n" % (sum_annotation_thumb) + \
            "        </a>\n" + \
            "      </td>\n"

    html += "      <td style=\"vertical-align: top\">\n" + \
            "        <a target=\"_blank\" href=\"%s\">S%03d Annotation Tiles<br/>\n" % (osum_annotation_img, slide_num) + \
            "          <img src=\"%s\" />\n" % (osum_annotation_thumb) + \
            "        </a>\n" + \
            "      </td>\n"

    html += "      <td style=\"vertical-align: top\">\n"
    if data_link:
      html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary\n" % slide_num + \
              "        (<a target=\"_blank\" href=\"%s\">Data</a>)</div>\n" % get_tile_annotation_path(slide_num)
    else:
      html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary</div>\n" % slide_num

    html += "        <div style=\"font-size: smaller; white-space: nowrap;\">\n" + \
            "          %s\n" % summary_annotation_stats(tile_summary).replace("\n", "<br/>\n          ") + \
            "        </div>\n" + \
            "      </td>\n"
    html += "    </tr>\n"
    return html

def generate_tiled_annotation_html_result(slide_nums, tile_summaries_dict, data_link):
    """
    Generate HTML to view the tiled images.
    Args:
      slide_nums: List of slide numbers.
      tile_summaries_dict: Dictionary of TileSummary objects keyed by slide number.
      data_link: If True, add link to tile data csv file.
    """
    slide_nums = sorted(slide_nums)
    if not TILE_ANNOTATION_SUMMARY_PAGINATE:
      html = ""
      html += html_header("Tiles Summary")

      html += "  <table>\n"
      for slide_num in slide_nums:
        html += image_row_annotation(slide_num, data_link)
      html += "  </table>\n"

      html += html_footer()
      text_file = open(os.path.join(TILE_ANNOTATION_SUMMARY_HTML_DIR, "tiles_annotation.html"), "w")
      text_file.write(html)
      text_file.close()
    else:
      total_len = len(slide_nums)
      page_size = TILE_ANNOTATION_SUMMARY_PAGINATION_SIZE
      num_pages = math.ceil(total_len / page_size)
      for page_num in range(1, num_pages + 1):
        start_index = (page_num - 1) * page_size
        end_index = (page_num * page_size) if (page_num < num_pages) else total_len
        page_slide_nums = slide_nums[start_index:end_index]

        html = ""
        html += html_header("Tiles Annotation Summary (Page %d)" % page_num)

        html += "  <div style=\"font-size: 20px\">"
        if page_num > 1:
          if page_num == 2:
            html += "<a href=\"tiles_annotation.html\">&lt;</a> "
          else:
            html += "<a href=\"tiles_annotation-%d.html\">&lt;</a> " % (page_num - 1)
        html += "Page %d" % page_num
        if page_num < num_pages:
          html += " <a href=\"tiles_annotation-%d.html\">&gt;</a> " % (page_num + 1)
        html += "</div>\n"

        html += "  <table>\n"
        for slide_num in page_slide_nums:
          tile_summary = tile_summaries_dict[slide_num]
          html += image_row_annotation(slide_num, tile_summary, data_link)
        html += "  </table>\n"

        html += html_footer()
        if page_num == 1:
          text_file = open(os.path.join(TILE_ANNOTATION_SUMMARY_HTML_DIR, "tiles_annotation.html"), "w")
        else:
          text_file = open(os.path.join(TILE_ANNOTATION_SUMMARY_HTML_DIR, "tiles_annotation-%d.html" % page_num), "w")
        text_file.write(html)
        text_file.close()

## 実行コード
multiprocess_annotation_to_tiles()

## スライド番号が途中から解析する場合
# start_ind = 50  #開始番号
# singleprocess_annotation_to_tiles(start_ind)