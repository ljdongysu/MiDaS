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

DATA_TYPE = ['kitti', 'dl', 'depth', 'server']

def get_file_name(file_name, image_path, dest_dir):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    image_list = []
    depth_list = []
    for image_name in file_lists:
        image_full_path = os.path.join(image_path, image_name)
        image_dest_path = os.path.join(dest_dir, image_name)
        image_dest_path = image_dest_path.replace(".jpg", ".png")
        image_list.append(image_full_path)
        depth_list.append(image_dest_path)

    return image_list, depth_list

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        pass

    return file_list

def GetImages(path, flag='kitti'):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        if os.path.exists(os.path.join(path, 'all.txt')):
            paths = [os.path.join(path, l.strip('\n').strip()) for l in open(os.path.join(path, 'all.txt')).readlines()]
        else:
            paths = Walk(path, ['jpg', 'jpeg', 'png', 'bmp', 'pfm'])
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    left_files, right_files = [], []
    if 'kitti' == flag:
        left_files = [f for f in paths if 'image_02' in f]
        right_files = [f.replace('/image_02/', '/image_03/') for f in left_files]
    elif 'dl' == flag:
        left_files = [f for f in paths if 'cam0' in f]
        right_files = [f.replace('/cam0/', '/cam1/') for f in left_files]
    elif 'depth' == flag:
        left_files = [f for f in paths if 'left' in f and 'disp' not in f]
        right_files = [f.replace('left/', 'right/').replace('left.', 'right.') for f in left_files]
    elif 'server' == flag:
        left_files = [f for f in paths if '.L' in f]
        right_files = [f.replace('L/', 'R/').replace('L.', 'R.') for f in left_files]
    else:
        raise Exception("Do not support mode: {}".format(flag))

    return left_files, right_files, root_len

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def write_depth(path, depth, grayscale, bits=1, input_path="", outpu_path=""):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    root_len = len(input_path)
    output_name = path[root_len+1:]

    depth_img_1740 = 1420.0 / (depth.copy()) * 256 * 256

    depth_img_1740_name = os.path.join(outpu_path, "scale_1740", output_name)
    depth_img_1740_name = depth_img_1740_name.replace(".jpg",".png")
    print(depth_img_1740_name, outpu_path)
    MkdirSimple(depth_img_1740_name)
    depth_img_1740[depth_img_1740 > 65535] = 65535
    depth_img_1740[depth_img_1740 < 0] = 65535
    cv2.imwrite(depth_img_1740_name, depth_img_1740.astype("uint16"))
    return

    depth_max_reduce = depth.copy()
    depth_max_reduce = np.max(depth_max_reduce) - depth_max_reduce

    scale_max_reduce = os.path.join(outpu_path, "scale_max_reduce", output_name)
    scale_max_reduce = scale_max_reduce.replace(".jpg",".png")
    MkdirSimple(scale_max_reduce)
    depth_max_reduce[depth_max_reduce > 65535] = 65535
    depth_max_reduce[depth_max_reduce < 0] = 0
    cv2.imwrite(scale_max_reduce, depth_max_reduce.astype("uint16"))


    max_min_scale = os.path.join(outpu_path, "original_demo", output_name)
    max_min_scale = max_min_scale.replace(".jpg",".png")
    MkdirSimple(max_min_scale)

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
    print("write depth image: {}".format(path + ".png"))
    if bits == 1:
        cv2.imwrite(max_min_scale, out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(max_min_scale, image_reduce.astype("uint16"))

    return

def run(img_list, dest_dir, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False, input_path=""):
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
    for image_file in img_list:
        print("image-file: {}".format(image_file))
        # input
        print(image_file)
        original_image_rgb = utils.read_image(image_file)  # in [0, 1]
        image = transform({"image": original_image_rgb})["image"]

        with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                 optimize, False)

        if not side:
            write_depth(image_file, prediction, grayscale, bits=2, input_path=input_path, outpu_path=dest_dir)

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)', required=True)

    parser.add_argument('--dest_dir',
                        help='Folder for save depth image', required=True)

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

    for k in DATA_TYPE:
        left_files, right_files, _ = GetImages(args.input_path, k)
        if len(left_files) != 0:
            break

    run(left_files, args.dest_dir, model_weights, args.model_type, grayscale=True, input_path=args.input_path)
