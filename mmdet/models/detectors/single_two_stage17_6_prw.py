import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

import torch.nn.functional as F
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core import bbox2result_reid

from ..roi_heads.bbox_heads.oim_nae_new import OIMLoss
from ..dense_heads.labeled_matching_layer_queue import LabeledMatchingLayerQueue
from ..dense_heads.unlabeled_matching_layer import UnlabeledMatchingLayer
from ..dense_heads.triplet_loss import TripletLossFilter
from ..utils import MINE

@DETECTORS.register_module()
class SingleTwoStageDetector176PRW(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleTwoStageDetector176PRW, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        self.loss_oim = OIMLoss()
        num_person = 5532
        queue_size = 5000
        self.labeled_matching_layer = LabeledMatchingLayerQueue(num_persons=num_person, feat_len=256) # for mot17half
        self.unlabeled_matching_layer = UnlabeledMatchingLayer(queue_size=queue_size, feat_len=256)
        self.loss_tri = TripletLossFilter()

        # self.mi_estimator = CLUBSample(256, 256, 512)
        # self.mi_estimator = CLUB()
        self.mi_estimator = MINE(256, 256, 256)
        # self.mi_estimator = L1OutUB(256, 256, 256)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleTwoStageDetector176PRW, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        xb = self.backbone(img)
        if self.with_neck:
            xn = self.neck(xb)
        #for xx in xb:
        #    print(xx.shape)
        #    print(xb[2].shape)
        return [xb[2]], xn

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        xb, xn = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(xb)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(xb, proposals)
        outs = outs + (roi_outs, )

        outs_n = self.bbox_head(xn)
        return outs, outs_n

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_ids,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        xb, xn = self.extract_feat(img)
        #print("here", xb.shape)

        losses = dict()

        # RPN forward and loss
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(xb)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas, cfg=proposal_cfg)
        else:
            proposal_list = proposals

        roi_losses, feats_pids_roi = self.roi_head.forward_train(xb, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, gt_ids,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        # oim_loss = dict()
        # oim_loss["loss_oim_roi"] = self.loss_oim(feats_pids_roi["bbox_feats"], feats_pids_roi["gt_pids"])
        # losses.update(oim_loss)


        single_losses, feats_pids = self.bbox_head.forward_train(xn, img_metas, gt_bboxes,
                                              gt_labels, gt_ids, gt_bboxes_ignore)
        # pos_reid = feats_pids["pos_reid"]
        # pos_reid_ids = feats_pids["pos_reid_ids"]
        # labeled_matching_scores, labeled_matching_reid, labeled_matching_ids = self.labeled_matching_layer(pos_reid, pos_reid_ids)
        # labeled_matching_scores *= 10
        # unlabeled_matching_scores = self.unlabeled_matching_layer(pos_reid, pos_reid_ids)
        # unlabeled_matching_scores *= 10
        # matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
        # pid_labels = pos_reid_ids.clone()
        # pid_labels[pid_labels == -2] = -1

        # p_i = F.softmax(matching_scores, dim=1)
        # focal_p_i = (1 - p_i)**2 * p_i.log()

        # #loss_oim = F.nll_loss(focal_p_i, pid_labels, reduction='none', ignore_index=-1)
        # loss_oim = F.nll_loss(focal_p_i, pid_labels, ignore_index=-1)

        # pos_reid = torch.cat((pos_reid, labeled_matching_reid), dim=0)
        # pid_labels = torch.cat((pid_labels, labeled_matching_ids), dim=0)
        # loss_tri = self.loss_tri(pos_reid, pid_labels)
        # single_losses["loss_oim_singel"] = loss_oim
        # single_losses["loss_tri"] = loss_tri

        ####### calculate mutual information ############
        feats_roi = feats_pids_roi["bbox_feats"]
        pids_roi = feats_pids_roi["gt_pids"]
        feats_fcos = feats_pids["pos_reid"]
        pids_fcos = feats_pids["pos_reid_ids"]
        dic1 = defaultdict(list)
        dic2 = defaultdict(list)

        for i in range(len(pids_roi)):
            if pids_roi[i] < 0:
                continue
            else:
                targets1_value = pids_roi[i].cpu().numpy().item()
                dic1[targets1_value].append(feats_roi[i])
        
        for i in range(len(pids_fcos)):
            if pids_fcos[i] < 0:
                continue
            else:
                targets2_value = pids_fcos[i].cpu().numpy().item()
                dic2[targets2_value].append(feats_fcos[i])
        
        all_feats1 = []
        all_feats2 = []
        for key, val in dic1.items():
            if key in dic2:
                val2 = dic2[key]
                feat1 = sum(val)/len(val)
                # print(feat1.shape)
                mean1 = F.normalize(feat1.unsqueeze(0))
                # mean1 = feat1.unsqueeze(0)
                feat2 = sum(val2)/len(val2)
                mean2 = F.normalize(feat2.unsqueeze(0))
                # mean2 = feat2.unsqueeze(0)
                all_feats1.append(mean1)
                all_feats2.append(mean2)

        if len(all_feats1) > 0 and len(all_feats2) >0:
            all_feats1 = torch.cat(all_feats1)
            all_feats2 = torch.cat(all_feats2)
            # print(all_feats1.shape, all_feats2.shape)

            all_feats1_d = all_feats1.detach()
            all_feats2_d = all_feats2.detach()
            mi_loss = dict()
            if torch.randint(1, 100, (1,)) % 3:
                self.mi_estimator.train()
                mi_loss["loss_mi"] = 0.2 * self.mi_estimator.learning_loss(all_feats1_d, all_feats2_d)
            else:
                self.mi_estimator.eval()
                # mi_loss["loss_mi_bound"] = self.mi_estimator(all_feats1, all_feats2)
                mi_loss["loss_mi_bound"] = 0.2 * self.mi_estimator.learning_loss(all_feats1, all_feats2)

            losses.update(mi_loss)

        # losses.update(single_losses)
        for key, val in single_losses.items():
            if key in losses:
                #print("losses", key, losses[key], losses[key].shape)
                #print("val", val, val.shape)
                losses[key] += val
            else:
                losses[key] = val

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        xb, xn = self.extract_feat(img)

        outs_n = self.bbox_head(xn, proposals)
        bbox_list = self.bbox_head.get_bboxes(
            *outs_n, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results_n = [
            bbox2result_reid(det_bboxes, det_labels, reid_feats, self.bbox_head.num_classes)
            for det_bboxes, det_labels, reid_feats in bbox_list
        ]

        proposals = []
        #print('scale_factor')
        #print(img_metas[0]['scale_factor'])
        tmp_sf = torch.FloatTensor(np.append(img_metas[0]['scale_factor'], 1)).to('cuda')
        #print(tmp_sf)
        for det_bboxes, det_labels, reid_feats in bbox_list:
            #print("det_bboxes")
            #print(det_bboxes.shape)
            #print(det_bboxes)
            proposals.append(torch.mul(det_bboxes, tmp_sf))
            

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(xb, img_metas)
        else:
            proposal_list = proposals
        #print("img_metas")
        #print(img_metas)
        #print("proposal_list")
        #print(proposal_list[0].shape)
        #print(proposal_list)

        bbox_results_b, det_features = self.roi_head.simple_test(
            xb, proposal_list, img_metas, rescale=rescale, use_rpn=False)

        #print("results_b")
        #print("det_features")
        #print(len(det_features))
        #print(det_features.shape)
        bbox_results_b = []
        #print("bbox_results_n")
        #print(bbox_results_n[0][0].shape)
        #print(bbox_results_n[0][0][:, 5:])
        bbox_results_b.append(bbox_results_n[0][0].copy())
        bbox_results_b[0][:, 5:] = det_features.cpu().numpy()
        #print(bbox_results_b[0][:, 5:])
        #print(bbox_results_n[0][0][:, 5:])
        
        #print(len(bbox_results_b))
        #print(bbox_results_b[0].shape)
        #print(bbox_results_b[0][:,:5])

        return bbox_results_n, bbox_results_b

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

