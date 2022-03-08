# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: refer: https://github.com/qwertyforce/image_search
"""
# !pip install ImageHash
# !pip install distance
# !pip install vptree
import os

import matplotlib.pyplot as plt


def show_images(images, figsize=(20, 10), columns=5):
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)
        # plt.show()


from PIL import Image
import imagehash
import numpy as np
import distance

IMAGE_PATH = '../examples/data/'
hashes = {}
file_names = os.listdir(IMAGE_PATH)
for file_name in file_names:
    phash = str(imagehash.phash(Image.open(f'{IMAGE_PATH}/{file_name}'), 16))
    if phash in hashes:
        hashes[phash].append(file_name)
    else:
        hashes[phash] = [file_name]

print(hashes)
query_image = Image.open(f'{IMAGE_PATH}/image1.jpeg')
query_image_phash = str(imagehash.phash(query_image, 16))
show_images([np.array(query_image)])

hamming_distances = []
for phash in hashes.keys():
    hamming_distances.append({"dist": distance.hamming(query_image_phash, phash), "phash": phash})
hamming_distances.sort(key=lambda item: item["dist"])
hamming_distances = hamming_distances[:10]

print(hamming_distances)
found_images = []
for it in hamming_distances:
    found_images.append(hashes[it["phash"]])
found_images = [item for sublist in found_images for item in sublist]
print('found_images:',found_images)
images_np = []
for image_filename in found_images:
    images_np.append(np.array(Image.open(f'{IMAGE_PATH}/{image_filename}')))



import vptree

tree = vptree.VPTree(list(hashes.keys()), distance.hamming)

neighbors = tree.get_n_nearest_neighbors(query_image_phash, 10)

print(neighbors)
vptree_found_images = []
for neighbor in neighbors:
    vptree_found_images.append(hashes[neighbor[1]])
vptree_found_images = [item for sublist in vptree_found_images for item in sublist]
print('vptree_found_images:',vptree_found_images)
images_np_vptree = []
for image_filename in vptree_found_images:
    images_np_vptree.append(np.array(Image.open(f'{IMAGE_PATH}/{image_filename}')))

show_images(images_np_vptree)

width, height = query_image.size
query_image_resized = query_image.resize((width // 19, height // 19))
print(distance.hamming(query_image_phash, str(imagehash.phash(query_image_resized, 16))))
show_images([np.array(query_image_resized)])

query_image_resized_2 = query_image.resize((width // 4, height // 23))
print(distance.hamming(query_image_phash, str(imagehash.phash(query_image_resized_2, 16))))
show_images([np.array(query_image_resized_2)])

crop_rectangle = (200, 200, 900, 900)
query_image_cropped = query_image.crop(crop_rectangle)
print(distance.hamming(query_image_phash, str(imagehash.phash(query_image_cropped, 16))))
show_images([np.array(query_image_cropped)])

query_image_rotated = query_image.rotate(180)
print(distance.hamming(query_image_phash, str(imagehash.phash(query_image_rotated, 16))))
show_images([np.array(query_image_rotated)])
