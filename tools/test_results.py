import cv2
import os
from scipy.io import loadmat
import os.path as osp
import numpy as np
import json
from PIL import Image
import pickle
import sys 

from sklearn.metrics import average_precision_score
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
    # change to your own path
    root_dir = '/home/yy1/2021/data/cuhk'

    with open(root_dir + '/annotation/test_new.json', 'r') as fid:
        test_det = json.load(fid)
    id_to_img = dict()
    img_to_id = dict()
    for td in test_det['images']:
        im_name = td['file_name'].split('/')[-1]
        im_id = td['id']
        id_to_img[im_id] = im_name
        img_to_id[im_name] = im_id

    # change to your own working dirs
    results_path = '/home/yy1/2021/AlignPS/work_dirs/' + sys.argv[1]
    #results_path = '/home/yy1/2021/mmdetection/work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_singlescale_focal_x4_bg-2_lconv3dcn_sub_triqueue'
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


    all_thresh = [0.2]
    #iou_thresh = 0.6
    #all_thresh = [0.05, 0.1, 0.15, 0.18, 0.2, 0.22, 0.25]
    #all_thresh = [0.15 + 0.01 * i for i in range(11)]
    for thresh in all_thresh:
        if db_name == "psdb_test":
            gallery_size= 100
            test = loadmat(osp.join(root_dir, "annotation/test/train_test/TestG{:d}.mat".format(gallery_size)))
            test = test["TestG{:d}".format(gallery_size)].squeeze()

            aps = []
            accs = []
            topk = [1, 5, 10]
            for index, item in enumerate(test):
                # query
                y_true, y_score = [], []
                count_gt, count_tp = 0, 0

                im_name = str(item["Query"][0, 0][0][0])
                query_gt_box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                query_gt_box[2:] += query_gt_box[:2]
                query_dict = gallery_dicts[img_to_id[im_name]]
                query_boxes = query_dict['bbox']
                iou, iou_max, nmax = get_max_iou(query_boxes, query_gt_box)
                #print(iou_max)
                '''
                if iou_max <= iou_thresh:
                    query_feat = query_dict['feats'][nmax]
                    #print("not detected", im_name, iou_max)
                    #continue
                else:
                    iou_good, good_idx = get_good_iou(query_boxes, query_gt_box, iou_thresh)
                    query_feats = query_dict['feats'][good_idx]
                    query_feat = iou_good[np.newaxis,:].dot(query_feats) / np.sum(iou_good)
                    query_feat = query_feat.ravel()
                '''

                query_feat = query_dict['feats'][nmax]
                query_feat = normalize(query_feat[np.newaxis,:], axis=1).ravel()

                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    gallery_imname = str(im_name[0])
                    gt = box[0].astype(np.int32)
                    count_gt += gt.size > 0
                    img_id = img_to_id[gallery_imname]
                    #if img_id not in gallery_dicts:
                    #    continue
                    det = np.asarray(gallery_dicts[img_id]['bbox'])
                    scores = np.asarray(gallery_dicts[img_id]['scores'])
                    keep_inds = np.where(scores >= thresh)
                    scores = scores[keep_inds]
                    det = det[keep_inds]

                    gallery_feat = gallery_dicts[img_id]['feats'][keep_inds]
                    if gallery_feat.shape[0] > 0:
                        gallery_feat = normalize(gallery_feat, axis=1)
                    else:
                        continue

                    sim = gallery_feat.dot(query_feat).ravel()
                    #Class Weighted Similarity
                    #print(scores)
                    #sim = sim * scores
                    label = np.zeros(len(sim), dtype=np.int32)

                    if gt.size > 0:
                        w, h = gt[2], gt[3]
                        gt[2:] += gt[:2]
                        iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                        inds = np.argsort(sim)[::-1]
                        sim = sim[inds]
                        det = det[inds]
                        # only set the first matched det as true positive
                        for j, roi in enumerate(det[:, :4]):
                            if compute_iou(roi, gt) >= iou_thresh:
                                label[j] = 1
                                count_tp += 1
                                break
                    y_true.extend(list(label))
                    y_score.extend(list(sim))


                y_score = np.asarray(y_score)
                y_true = np.asarray(y_true)
                assert count_tp <= count_gt
                recall_rate = count_tp * 1.0 / count_gt
                ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
                aps.append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true = y_true[inds]
                accs.append([min(1, sum(y_true[:k])) for k in topk])


            print("threshold: ", thresh)
            print("  mAP = {:.2%}".format(np.mean(aps)))
            accs = np.mean(accs, axis=0)
            for i, k in enumerate(topk):
                print("  Top-{:2d} = {:.2%}".format(k, accs[i]))



