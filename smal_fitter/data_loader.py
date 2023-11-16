import sys

sys.path.append('../')
import numpy as np
import cv2
import torch
import imageio
from tqdm import tqdm
from utils import crop_to_silhouette
import os
import json

from csv import DictReader
from PIL import Image, ImageFilter
from pycocotools.mask import decode as decode_RLE
from copy import copy

import config


def load_badja_sequence(BADJA_PATH, sequence_name, crop_size, image_range=None):
    file_names = []
    rgb_imgs = []
    sil_imgs = []
    joints = []
    visibility = []

    annotations_path = os.path.join(BADJA_PATH, "joint_annotations")
    json_path = os.path.join(annotations_path, "{0}.json".format(sequence_name))
    with open(json_path) as json_data:
        sequence_annotation = np.array(json.load(json_data))
        if image_range is not None:
            sequence_annotation = sequence_annotation[image_range]
        for image_annotation in tqdm(sequence_annotation):
            file_name = os.path.join(BADJA_PATH, image_annotation['image_path'])
            seg_name = os.path.join(BADJA_PATH, image_annotation['segmentation_path'])

            if os.path.exists(file_name) and os.path.exists(seg_name):
                landmarks = np.array(image_annotation['joints'])[config.BADJA_ANNOTATED_CLASSES]
                visibility.append(np.array(image_annotation['visibility'])[config.BADJA_ANNOTATED_CLASSES])

                rgb_img = imageio.imread(file_name) / 255.0
                sil_img = imageio.imread(seg_name)[:, :, 0] / 255.0

                rgb_h, rgb_w, _ = rgb_img.shape
                sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)

                sil_img, rgb_img, landmarks = crop_to_silhouette(sil_img, rgb_img, landmarks, crop_size)

                rgb_imgs.append(rgb_img)
                sil_imgs.append(sil_img)
                joints.append(landmarks)
                file_names.append(os.path.basename(image_annotation['image_path']))

            elif os.path.exists(file_name):
                print("BADJA SEGMENTATION file path: {0} is missing".format(seg_name))
            else:
                print("BADJA IMAGE file path: {0} is missing".format(file_name))

    rgb = torch.FloatTensor(np.stack(rgb_imgs, axis=0)).permute(0, 3, 1, 2)
    sil = torch.FloatTensor(np.stack(sil_imgs, axis=0))[:, None, :, :]
    joints = torch.FloatTensor(np.stack(joints, axis=0))
    visibility = torch.FloatTensor(np.stack(visibility, axis=0).astype(np.float))

    ## Sets invalid joints (i.e. not labelled) as invisible
    invalid_joints = np.array(config.BADJA_ANNOTATED_CLASSES) == -1
    visibility[:, invalid_joints] = 0.0

    return (rgb, sil, joints, visibility), file_names


def load_stanford_sequence(STANFORD_EXTRA, image_name, crop_size):
    # edit this to the location of the extracted StanfordDogs tar file (e.g. /.../Images).
    img_dir = os.path.join(STANFORD_EXTRA, "datasets")

    # edit this to the location of the downloaded full dataset .json
    json_loc = os.path.join(STANFORD_EXTRA + f"/datasets/SE_{image_name}.json")

    # load json into memory
    with open(json_loc) as infile:
        json_data = json.load(infile)
        json_data[0]['img_path'] = os.path.join(STANFORD_EXTRA, f"datasets/{image_name}.jpg")

    # convert json data to a dictionary of img_path : all_data, for easy lookup
    # json_dict = {i['img_path']: i for i in json_data}
    json_dict = {image_name: i for i in json_data}

    def get_seg_from_entry(entry):
        """Given a .json entry, returns the binary mask as a numpy array"""

        rle = {
            "size": [entry['img_height'], entry['img_width']],
            "counts": entry['seg']
        }

        decoded = decode_RLE(rle)
        return decoded

    def get_dog(name):
        data = json_dict[name]

        # load img
        img_data = imageio.imread(os.path.join(img_dir, data['img_path']))

        # load seg
        # seg_data = get_seg_from_entry(data)
        mask_dir = os.path.join(STANFORD_EXTRA, f"datasets/masks/{image_name}.jpg")
        seg_data = cv2.imread(mask_dir)
        # seg_data = np.array(data['seg'])

        # add to output
        data['img_data'] = img_data
        data['seg_data'] = seg_data[:, :, 0]

        return data

    loaded_data = get_dog(image_name)
    '''(25,3) 节点'''
    # add an extra dummy invisble joint for tail_mid which wasn't annotated in Stanford-Extra
    # raw_joints = np.concatenate([
    #     np.array(loaded_data['joints']), [[0.0, 0.0, 0.0]]], axis = 0)
    raw_joints = np.array(loaded_data['joints'])

    sil_img, rgb_img, landmarks = crop_to_silhouette(
        loaded_data['seg_data'], loaded_data['img_data'][:, :, :3] / 255.0,
        raw_joints[:, [1, 0]], crop_size)

    rgb = torch.FloatTensor(rgb_img)[None, ...].permute(0, 3, 1, 2)
    sil = torch.FloatTensor(sil_img)[None, None, ...]
    joints = torch.FloatTensor(landmarks)[:, :2].unsqueeze(0)
    visibility = torch.FloatTensor(raw_joints)[:, -1].unsqueeze(0)
    file_names = [os.path.basename(loaded_data['img_path'])]

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,8))
    # plt.imshow(sil[0][0])
    # plt.show()

    # import cv2
    # img0 = np.array(rgb[0].permute(1,2,0))
    # for i, joint in enumerate(joints[0]):
    #     [x, y] = joint.tolist()
    #     cv2.circle(img0, (int(y), int(x)), 1, (255,0,255), -1)
    #     cv2.putText(img0, f"{i}", (int(y+1), int(x)),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0))
    # cv2.namedWindow('win', 0)
    # cv2.resizeWindow('win', 1000, 800)
    # cv2.imwrite("/media/scau2311/A/xcg/SMALify/data/PIG_DATA/sample_imgs/vis_joints.jpg", img0[:,:,::-1])
    # cv2.imshow('win', img0)
    # cv2.waitKey(5000)

    return (rgb, sil, joints, visibility), file_names


