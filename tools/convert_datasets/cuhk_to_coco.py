import json
import os
import cv2
import pickle
import numpy as np

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = '/Users/yichaoyan/Documents/PAMI-context/cuhk-sysu-group/dataset'
cache_file = '/Users/yichaoyan/Documents/PAMI-context/cuhk-sysu-group/dataset/annotation/cache/psdb_train_roidb.pkl'
#cache_file = '/Users/yichaoyan/Documents/PAMI-context/cuhk-sysu-group/dataset/annotation/cache/psdb_test_roidb.pkl'
# 用于创建训练集或验证集

# 训练集和验证集划分的界线
dataset = dict()
dataset['categories'] = []
dataset['images'] = []
dataset['annotations'] = []
classes = ['__background__', 'person']

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes):
    if cls == '__background__':
        continue
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'object'})

with open(cache_file, "rb") as f:
    roidb = pickle.load(f)

#phase = 'trainval'
phase = 'train_pid'
#phase = 'val'
#phase = 'val_pid'
#phase = 'test_det_pid'
#split = 10000
#if phase == "train":
#    roidb = roidb[:split]
#elif phase == 'val':
#    roidb = roidb[split:]
#roidb = roidb[split:]
id = 0
for k, anno in enumerate(roidb):
    dataset['images'].append({'file_name': anno["image"].replace('/Users/yichaoyan/Documents/PAMI-context/cuhk-sysu-group/dataset', '/raid/yy1/data/cuhk'),
                              'id': k,
                              'width': anno["width"],
                              'height': anno["height"]})

    for box, pid in zip(anno["gt_boxes"],  anno["gt_pids"]):
        bwidth = box[2] - box[0]
        bheight = box[3] - box[1]
        #if pid == -1:
        #    continue
        dataset['annotations'].append({
            'area': int(bwidth * bheight),
            'bbox': np.array([box[0], box[1], bwidth, bheight]).tolist(),
            'category_id': int(1),
            'id': int(id),
            'image_id': k,
            'iscrowd': 0,
            'person_id': int(pid),
            # mask, 矩形是从左上角点按顺时针的四个顶点
            'segmentation': []
        })
        id += 1

json_name = os.path.join(root_path, 'annotation/{}.json'.format(phase))
with open(json_name, 'w') as f:
  json.dump(dataset, f)

