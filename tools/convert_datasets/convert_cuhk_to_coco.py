import traceback
import argparse
import datetime
import json
import cv2
import os
from scipy.io import loadmat
import os.path as osp
import numpy as np
from PIL import Image
import pickle


def set_box_pid(boxes, box, pids, pid):
    for i in range(boxes.shape[0]):
        if np.all(boxes[i] == box):
            pids[i] = pid
            return
    print("Person: %s, box: %s cannot find in images." % (pid, box))

def image_path_at(data_path, image_index, i):
    image_path = osp.join(data_path, image_index[i])
    assert osp.isfile(image_path), "Path does not exist: %s" % image_path
    return image_path

def load_image_index(root_dir, db_name):
    """Load the image indexes for training / testing."""
    # Test images
    test = loadmat(osp.join(root_dir, "annotation", "pool.mat"))
    test = test["pool"].squeeze()
    test = [str(a[0]) for a in test]
    if db_name == "psdb_test":
        return test

    # All images
    all_imgs = loadmat(osp.join(root_dir, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    all_imgs = [str(a[0][0]) for a in all_imgs]

    # Training images = all images - test images
    train = list(set(all_imgs) - set(test))
    train.sort()
    return train

if __name__ == "__main__":
    db_name = "psdb_test"
    root_dir = '/Users/yichaoyan/Documents/PAMI-context/cuhk-sysu-group/dataset'
    cache_path = osp.join(root_dir, "annotation", "cache")
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_file = osp.join(cache_path, db_name + "_roidb.pkl")
    data_path = osp.join(root_dir, "Image", "SSM")
    image_index = load_image_index(root_dir, db_name)
    all_imgs = loadmat(osp.join(root_dir, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    name_to_boxes = {}
    name_to_pids = {}
    for im_name, _, boxes in all_imgs:
        im_name = str(im_name[0])
        boxes = np.asarray([b[0] for b in boxes[0]])
        boxes = boxes.reshape(boxes.shape[0], 4)
        valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
        assert valid_index.size > 0, "Warning: %s has no valid boxes." % im_name
        boxes = boxes[valid_index]
        name_to_boxes[im_name] = boxes.astype(np.int32)
        name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

    if db_name == "psdb_train":
        train = loadmat(osp.join(root_dir, "annotation/test/train_test/Train.mat"))
        train = train["Train"].squeeze()
        for index, item in enumerate(train):
            scenes = item[0, 0][2].squeeze()
            for im_name, box, _ in scenes:
                im_name = str(im_name[0])
                box = box.squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)

        roidb = []
        for i, im_name in enumerate(image_index):
            boxes = name_to_boxes[im_name]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            pids = name_to_pids[im_name]
            size = Image.open(image_path_at(data_path, image_index, i)).size
            roidb.append(
                {
                    "gt_boxes": boxes,
                    "gt_pids": pids,
                    "image": image_path_at(data_path, image_index, i),
                    "height": size[1],
                    "width": size[0],
                    "flipped": False,
                }
            )
        with open(cache_file, "wb") as f:
            pickle.dump(roidb, f)

    if db_name == "psdb_test":
        test = loadmat(osp.join(root_dir, "annotation/test/train_test/TestG50.mat"))
        test = test["TestG50"].squeeze()
        for index, item in enumerate(test):
            # query
            im_name = str(item["Query"][0, 0][0][0])
            box = item["Query"][0, 0][1].squeeze().astype(np.int32)
            set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)

            # gallery
            gallery = item["Gallery"].squeeze()
            for im_name, box, _ in gallery:
                im_name = str(im_name[0])
                if box.size == 0:
                    break
                box = box.squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)

        roidb = []
        for i, im_name in enumerate(image_index):
            boxes = name_to_boxes[im_name]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            pids = name_to_pids[im_name]
            size = Image.open(image_path_at(data_path, image_index, i)).size
            roidb.append(
                {
                    "gt_boxes": boxes,
                    "gt_pids": pids,
                    "image": image_path_at(data_path, image_index, i),
                    "height": size[1],
                    "width": size[0],
                    "flipped": False,
                }
            )
        with open(cache_file, "wb") as f:
            pickle.dump(roidb, f)