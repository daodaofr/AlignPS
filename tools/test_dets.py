import cv2
import os
from scipy.io import loadmat
import os.path as osp
import numpy as np
import json
from PIL import Image
import pickle

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import normalize

from iou_utils import get_max_iou, get_good_iou

def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

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
    root_dir = '/raid/yy1/data/cuhk'

    with open('/raid/yy1/data/cuhk/annotation/test.json', 'r') as fid:
        test_det = json.load(fid)
    id_to_img = dict()
    img_to_id = dict()
    gt_box_dict = dict()
    for td in test_det['images']:
        im_name = td['file_name'].split('/')[-1]
        im_id = td['id']
        id_to_img[im_id] = im_name
        img_to_id[im_name] = im_id
    
    for ann in test_det['annotations']:
        if ann['image_id'] not in gt_box_dict:
            gt_box_dict[ann['image_id']] = []
        tmp_box = ann['bbox']
        gt_box_dict[ann['image_id']].append([tmp_box[0], tmp_box[1], tmp_box[0] + tmp_box[2], tmp_box[1]+tmp_box[3]])

    results_path = '/raid/yy1/mmdetection/work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue_t3_dcn0'
    #results_path = '/raid/yy1/mmdetection/work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_fpncat'
    with open(os.path.join(results_path, 'results_1000.pkl'), 'rb') as fid:
        all_dets = pickle.load(fid)

    gallery_dicts = {}
    for i, dets in enumerate(all_dets):
        image_id = i
        gallery_dicts[image_id] = dict()
        gallery_dicts[image_id]['bbox'] = dets[0][:, :4]
        gallery_dicts[image_id]['scores'] = dets[0][:, 4]
        gallery_dicts[image_id]['feats'] = dets[0][:, 5:]


    thresh = 0.2
    iou_thresh=0.5
    #iou_thresh = 0.6
    #all_thresh = [0.05, 0.1, 0.15, 0.18, 0.2, 0.22, 0.25]
    #all_thresh = [0.15 + 0.01 * i for i in range(11)]
    y_true, y_score = [], []
    count_gt, count_tp = 0, 0

    for key, gt_boxes in gt_box_dict.items():
        gt_boxes = np.asarray(gt_boxes)
        det = gallery_dicts[key]['bbox']
        scores = gallery_dicts[key]['scores']
        if det.shape[0] > 0 :
            det = np.asarray(det)
            inds = np.where(scores.ravel() >= thresh)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
        else:
            num_det = 0
        if num_det == 0:
            count_gt += num_gt
            continue

        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = compute_iou(gt_boxes[i], det[j, :4])

        tfmat = (ious >= iou_thresh)
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False
        for j in range(num_det):
            y_score.append(det[j, -1])
            if tfmat[:, j].any():
                y_true.append(True)
            else:
                y_true.append(False)
        count_tp += tfmat.sum()
        count_gt += num_gt

    det_rate = count_tp * 1.0 / count_gt
    ap = average_precision_score(y_true, y_score) * det_rate
    precision, recall, __ = precision_recall_curve(y_true, y_score)
    recall *= det_rate

    print('  recall = {:.2%}'.format(det_rate))
    print('  ap = {:.2%}'.format(ap))




            



