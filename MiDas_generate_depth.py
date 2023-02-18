import os
import glob
import torch
import utils
import cv2
import argparse
import time

import numpy as np

from run import process

from run import create_side_by_side

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

first_execution = True

from tqdm.contrib import tzip

def get_file_name(file_name, image_path):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    image_list = []
    depth_list = []
    for image_name in file_lists:
        image_full_path = os.path.join(image_path, image_name)
        image_dest_path = image_full_path.replace("REMAP", "DEPTH/MiDas").split()[0]
        image_dest_path = image_dest_path.replace(".jpg", ".png")
        image_list.append(image_full_path)
        depth_list.append(image_dest_path)

    return image_list, depth_list

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    image_reduce = max_val - out.astype("uint16")

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", image_reduce.astype("uint16"))

    return

def run(img_list, depth_list, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    # get input
    for image_file, depth_image_file in tzip(img_list, depth_list):

        # input
        print(image_file.split()[0])
        original_image_rgb = utils.read_image(image_file.split()[0])  # in [0, 1]
        image = transform({"image": original_image_rgb})["image"]

        with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                 optimize, False)

        MkdirSimple(depth_image_file)

        depth_image_file = depth_image_file.replace(".jpg","")

        if not side:
            write_depth(depth_image_file, prediction, grayscale, bits=2)

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument("--file_name", help='image name list', required=True)

    parser.add_argument('-t', '--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    args = parser.parse_args()

    model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    img_list, depth_list = get_file_name(args.file_name, args.input_path)

    run(img_list, depth_list, model_weights, args.model_type, grayscale=True)
