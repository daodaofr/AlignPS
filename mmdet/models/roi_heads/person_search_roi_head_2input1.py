from mmdet.core import bbox2result, bbox2roi, bbox2result_reid
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

import torch
from torch import nn
import torch.nn.functional as F

@HEADS.register_module()
class PersonSearchRoIHead2Input1(StandardRoIHead):
    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_pids,
        gt_bboxes_ignore=None,
        gt_masks=None,
    ):
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

        losses = dict()
        feats_pids = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, gt_pids, img_metas
            )
            losses.update(bbox_results["loss_bbox"])
            feats_pids['bbox_feats'] = bbox_results["feature"]
            feats_pids['gt_pids'] = bbox_results["gt_pids"]

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(
                x, sampling_results, bbox_results["bbox_feats"], gt_masks, img_metas
            )
            # TODO: Support empty tensor input. #2280
            if mask_results["loss_mask"] is not None:
                losses.update(mask_results["loss_mask"])

        return losses, feats_pids

    def _bbox_forward(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(x[: self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats1 = F.adaptive_max_pool2d(bbox_feats, 1)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            bbox_feats = F.adaptive_max_pool2d(bbox_feats, 1)
        cls_score, bbox_pred, feature = self.bbox_head(bbox_feats1, bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, feature=feature, bbox_feats=bbox_feats
        )
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, gt_pids, img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, gt_pids, self.train_cfg
        )
        loss_bbox = self.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            bbox_results["feature"],
            rois,
            *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(gt_pids=bbox_targets[0])
        return bbox_results

    def simple_test_bboxes(
        self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, use_rpn=True
    ):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        if not use_rpn:
            return None, None, bbox_results["feature"]
        img_shape = img_metas[0]["img_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        det_bboxes, det_labels, det_features = self.bbox_head.get_bboxes(
            rois,
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            bbox_results["feature"],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg,
        )
        return det_bboxes, det_labels, det_features

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False, use_rpn=True):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        det_bboxes, det_labels, det_features = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, use_rpn=use_rpn
        )
        if det_bboxes is None:
            return None, det_features
        #bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        bbox_results = bbox2result_reid(det_bboxes, det_labels, det_features, self.bbox_head.num_classes)
        #bbox_results = [
        #    bbox2result_reid(det_bboxes[i], det_labels[i], det_features[i],
        #                self.bbox_head.num_classes)
        #    for i in range(len(det_bboxes))
        #]
        
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale
            )
            return bbox_results, segm_results
