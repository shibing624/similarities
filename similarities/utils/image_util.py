# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import base64
import sys
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image
from loguru import logger
from tqdm import tqdm


def is_link(s):
    return s is not None and s.startswith('http')


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)


def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        image_file = img
        with open(image_file, 'rb') as f:
            img_str = f.read()
            img = img_decode(img_str)
        if img is None:
            try:
                buf = BytesIO()
                image = BytesIO(img_str)
                im = Image.open(image)
                rgb = im.convert('RGB')
                rgb.save(buf, 'jpeg')
                buf.seek(0)
                image_bytes = buf.read()
                data_base64 = str(base64.b64encode(image_bytes),
                                  encoding="utf-8")
                image_decode = base64.b64decode(data_base64)
                img_array = np.frombuffer(image_decode, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def preprocess_image(img, alpha_color=(255, 255, 255)):
    """
    preprocess image
    :param img:
    :param alpha_color:
    :return:
    """
    img = check_img(img)
    if img is None:
        return None
    img = alpha_to_color(img, alpha_color)
    return img
