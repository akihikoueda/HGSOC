import os
import numpy as np
from PIL import Image, ImageDraw
import tqdm
import silence_tensorflow.auto
import tensorflow as tf

## settings ##
BATCH_SIZE = 2
TILE_EXPANTION = 1

# HE image normalization parameters
normalize_Io = 240
normalize_alpha = 1
normalize_beta = 0.15

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def preprocessimg(img, microns_per_pixel, export_tile_scale, row_tile_size, col_tile_size):
    w, h = img.size
    new_w = round(w * microns_per_pixel / 0.25 / export_tile_scale * TILE_EXPANTION)
    new_h = round(h * microns_per_pixel / 0.25 / export_tile_scale * TILE_EXPANTION)
    cropped_img = crop_center(img.resize((new_w, new_h)), row_tile_size * TILE_EXPANTION, col_tile_size * TILE_EXPANTION)
    np_img = np.array(cropped_img)
    try:
        np_img = normalizeStaining(np_img)
    except:
        np_img = np_img
    return np_img

def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
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
    return Inorm


def annotate_label(model, images_list):
    images_array = np.array(images_list).astype(np.float32)
    predction_list = model.predict(images_array/255, batch_size = BATCH_SIZE, verbose=1)
    label_lists = list()
    pred_lists = list()
    for i in range(len(predction_list)):
        class_label = np.argmax(predction_list[i])
        label_lists.append(class_label)
        pred_prob = np.max(predction_list[i])
        pred_lists.append(pred_prob)
    return label_lists, pred_lists


def generate_tile_annotation_summaries(df, summary_pil_img, tile_size, case_id, working_dir, mode):
    """
    Generate summary images showing a 'heatmap' representation of the tissue segmentation of all tiles.
    Args:
      tile_sum: TileSummary object.
      np_orig: Image as a NumPy array of original and filtered result image, respectively.
    """
    draw_img = ImageDraw.Draw(summary_pil_img)
    df = df.loc[:,['row', 'col', 'pattern_label', 'pattern_prob', 'tils_label', 'tils_prob']]
    for _, data in df.iterrows():
        r, c, pattern_label, pattern_prob, tils_label, tils_prob = data.to_list()
        if mode == 'pattern':
            color = tile_annotation_border_color(pattern_label, pattern_prob)
            tile_border(draw_img, (r-1)*tile_size, r*tile_size, (c-1)*tile_size, c*tile_size, color)
        if mode == 'tils':
            color = tile_annotation_border_color(tils_label, tils_prob) 
            tile_border(draw_img, (r-1)*tile_size, r*tile_size, (c-1)*tile_size, c*tile_size, color)
    export_filepath = get_annotation_summary_image_path(case_id, working_dir, mode)
    return summary_pil_img, export_filepath


def tile_border(draw, r_s, r_e, c_s, c_e, color):
    """
    Draw a border around a tile with width border size.
    Args:
      draw: Draw object for drawing on PIL image.
      r_s: Row starting pixel.
      r_e: Row ending pixel.
      c_s: Column starting pixel.
      c_e: Column ending pixel.
      color: Color of the border.
      border_size: Width of tile border in pixels.
    """
    draw.rectangle([(c_s, r_s), (c_e - 1, r_e - 1)], fill=color)

    
def tile_annotation_border_color(tissue_label, prob):
    """
    Obtain the corresponding tile border color for a particular tile tissue label.
    Args:
      tissue_label: The tile tissue label
    Returns:
      The tile border color corresponding to the tile tissue label.
    """
    if tissue_label == 'MT':
        border_color = (int(255*prob),0,0) # Red
    elif tissue_label == 'PG':
        border_color = (0,int(255*prob),0) # Green
    elif tissue_label == 'SP':
        border_color = (0,0,int(255*prob)) # Blue
    elif tissue_label == 'UD':
        border_color = (int(255*prob),int(255*prob),int(255*prob)) # White
    elif tissue_label == 1.0:
        border_color = (int(255*prob), 0, int(255*prob)) # Purple
    else:
        border_color = (0,0,0)
    return border_color


def get_annotation_summary_image_filename(case_id, mode):
    """
    Convert slide number to an annotation tile summary image file name.
    Example:
      HGSOC_005-tile_summary.png
    Args:
      case_name: The name of the case.
    Returns:
      The tile annotation summary image file name.
    """
    ext = 'jpg'
    img_filename = case_id + '_' + 'annotation' + '_' + mode + '.' + ext
    return img_filename


def get_annotation_summary_image_path(case_id, working_dir, mode):
    TILE_ANNOTATION_DIR = os.path.join(working_dir , 'annotation' + '_' + mode)
    os.makedirs(TILE_ANNOTATION_DIR, exist_ok=True)
    img_path = os.path.join(TILE_ANNOTATION_DIR, get_annotation_summary_image_filename(case_id, mode))
    return img_path