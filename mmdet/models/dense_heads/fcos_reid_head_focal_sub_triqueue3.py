import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multiclass_nms_reid
from ..builder import HEADS, build_loss
from .anchor_free_head_reid import AnchorFreeHeadReid
from .labeled_matching_layer_queue import LabeledMatchingLayerQueue
from .unlabeled_matching_layer import UnlabeledMatchingLayer
from .triplet_loss import TripletLossFilter

INF = 1e8


@HEADS.register_module()
class FCOSReidHeadFocalSubTriQueue3(AnchorFreeHeadReid):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 #regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                 #                (512, INF)),
                 regress_ranges=((-1, INF), (-2, -1), (-2, -1), (-2, -1),
                                (-2, -1)),
                 #regress_ranges=((-1, INF), (-2, -1), (-2, -1)),
                 #regress_ranges=((-1, 128), (128, INF), (-2, -1), (-2, -1),
                 #               (-2, -1)),
                 #regress_ranges=((-1, INF),),
                 #regress_ranges=((-2, -1), (-1, INF), (-2, -1), (-2, -1),
                 #               (-2, -1)),
                 #regress_ranges=((-2, -1), (-2, -1), (-1, INF), (-2, -1),
                 #               (-2, -1)),
                 #regress_ranges=((-1, 128), (128, INF), (-2, -1), (-2, -1),
                 #               (-2, -1)),
                 #regress_ranges=((-1, 128), (128, 256), (256, INF), (-2, -1),
                 #               (-2, -1)),
                 #regress_ranges=((-2, -1), (-1, 256), (256, INF), (-2, -1),
                 #               (-2, -1)),
                 #regress_ranges=((-1, INF), (-1, INF), (-1, INF), (-1, INF),
                 #                (-1, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.background_id = -2

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_tri = TripletLossFilter()

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        #self._init_reid_convs()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        #self.conv_reid = nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1)
        # num_person = 483
        num_person = 5532
        # queue_size = 500
        queue_size = 5000
        #self.classifier_reid = nn.Linear(self.feat_channels, num_person)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.labeled_matching_layer = LabeledMatchingLayerQueue(num_persons=num_person, feat_len=self.in_channels) # for mot17half
        self.unlabeled_matching_layer = UnlabeledMatchingLayer(queue_size=queue_size, feat_len=self.in_channels)

    def _init_reid_convs(self):
        """Initialize classification conv layers of the head."""
        self.reid_convs = nn.ModuleList()
        #for i in range(self.stacked_convs):
        for i in range(1):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reid_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    #norm_cfg=self.norm_cfg,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    bias=self.conv_bias))

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)
        #normal_init(self.conv_reid, std=0.01)
        #for m in self.reid_convs:
        #    if isinstance(m.conv, nn.Conv2d):
        #        normal_init(m.conv, std=0.01)

    def forward(self, feats, proposals=None):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        #print(len(feats), self.scales, self.strides)
        #print(len(tuple([feats[0]])), nn.ModuleList([self.scales[0]]), [self.strides[0]])
        #for single stage prediction
        #return multi_apply(self.forward_single, tuple([feats[0]]), nn.ModuleList([self.scales[0]]),
        #                    [self.strides[0]])
        feats = list(feats)
        h, w = feats[0].shape[2], feats[0].shape[3]
        mean_value = nn.functional.adaptive_avg_pool2d(feats[0], 1)
        mean_value = F.upsample(input=mean_value, size=(h, w), mode='bilinear')
        feats[0] = feats[0] - mean_value
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        #print(x.shape)
        #print('feat shape: ', x.shape, 'stride: ', stride)
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        
        reid_feat = x
        #for reid_layer in self.reid_convs:
        #    reid_feat = reid_layer(reid_feat)
        #reid_feat = self.conv_reid(reid_feat)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness, reid_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'reid_feat'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             reid_feats,
             gt_bboxes,
             gt_labels,
             gt_ids, 
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(reid_feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, ids, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels, gt_ids)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_reid = [
            reid_feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels)
            for reid_feat in reid_feats
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_reid = torch.cat(flatten_reid)
        #print("flatten reid", flatten_reid.shape)
        flatten_labels = torch.cat(labels)
        flatten_ids = torch.cat(ids)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        #pos_inds = nonzero((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        # background index
        '''
        bg_inds = ((flatten_labels < 0)
                    | (flatten_labels == bg_class_ind)).nonzero().reshape(-1)
        num_bg = len(bg_inds)
        bg_cls_scores = flatten_cls_scores[bg_inds]
        if num_bg > num_pos:
            cls_ids = torch.argsort(bg_cls_scores.squeeze(), descending=True)
            bg_inds = bg_inds[cls_ids[:num_pos]]
        '''

        pos_reid = flatten_reid[pos_inds]
        #bg_reid = flatten_reid[bg_inds]
        #pos_reid = torch.cat((pos_reid, bg_reid))
        # pos_reid_o = pos_reid.clone()
        pos_reid = F.normalize(pos_reid)


        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

            
            pos_reid_ids = flatten_ids[pos_inds]
            #bg_reid_ids = flatten_ids[bg_inds]
            #pos_reid_ids = torch.cat((pos_reid_ids, bg_reid_ids))
            #loss_oim = self.loss_reid(pos_reid, pos_reid_ids)
            #print(pos_reid.shape, pos_reid_ids.shape)
            #print(pos_reid_ids)
            
            # reid oim loss
            labeled_matching_scores, labeled_matching_reid, labeled_matching_ids = self.labeled_matching_layer(pos_reid, pos_reid_ids)
            labeled_matching_scores *= 10
            unlabeled_matching_scores = self.unlabeled_matching_layer(pos_reid, pos_reid_ids)
            unlabeled_matching_scores *= 10
            matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
            pid_labels = pos_reid_ids.clone()
            pid_labels[pid_labels == -2] = -1

            p_i = F.softmax(matching_scores, dim=1)
            #focal_p_i = 0.25 * (1 - p_i)**2 * p_i.log()
            focal_p_i = (1 - p_i)**2 * p_i.log()
            #focal_p_i = 2*(1 - p_i)**2 * p_i.log()
            #focal_p_i = 0.75*(1 - p_i)**2 * p_i.log()
            #focal_p_i = 1.25*(1 - p_i)**2 * p_i.log()
            #focal_p_i = 0.5*(1 - p_i)**2 * p_i.log()

            #loss_oim = F.nll_loss(focal_p_i, pid_labels, reduction='none', ignore_index=-1)
            loss_oim = F.nll_loss(focal_p_i, pid_labels, ignore_index=-1)

            pos_reid1 = torch.cat((pos_reid, labeled_matching_reid), dim=0)
            pid_labels1 = torch.cat((pid_labels, labeled_matching_ids), dim=0)
            loss_tri = self.loss_tri(pos_reid1, pid_labels1)
            #loss_oim = F.cross_entropy(matching_scores, pid_labels, ignore_index=-1)
            '''
            # softmax 
            matching_scores = self.classifier_reid(pos_reid).contiguous()
            loss_oim = F.cross_entropy(matching_scores, pos_reid_ids, ignore_index=-1)
            '''

            
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_oim = pos_reid.sum()
            loss_tri = pos_reid.sum()
            print('no gt box')

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_oim=loss_oim,
            loss_tri=loss_tri), dict(pos_reid=pos_reid, pos_reid_ids=pos_reid_ids, out_preds=p_i)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'reid_feats'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   reid_feats,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(reid_feats)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            reid_feat_list = [
                reid_feats[i][img_id].detach() for i in range(num_levels)
            ]
            #print(img_metas)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 centerness_pred_list,
                                                 reid_feat_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           reid_feats,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5 + dim), where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1, dim is the reid feature dimension.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points) == len(reid_feats)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_reid_feats = []
        for cls_score, bbox_pred, centerness, points, reid_feat in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points, reid_feats):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            #reid_feat = reid_feat.permute(1, 2, 0).reshape(-1, 256)
            reid_feat = reid_feat.permute(1, 2, 0).reshape(-1, self.in_channels)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                reid_feat = reid_feat[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_reid_feats.append(reid_feat)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_reid_feats = torch.cat(mlvl_reid_feats)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels, det_reid_feats = multiclass_nms_reid(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_reid_feats,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels, det_reid_feats

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, gt_ids_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        #for single stage prediction
        #points = [points[0]]
        #print(points, self.regress_ranges)
        #print(len(points), len(self.regress_ranges))

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, ids_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_ids_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        ids_list = [ids.split(num_points, 0) for ids in ids_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_ids = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_ids.append(
                torch.cat([ids[i] for ids in ids_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_ids, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_ids, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_ids.new_full((num_points,), self.background_id), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        ids = gt_ids[min_area_inds]
        labels[min_area == INF] = self.background_label  # set as BG
        ids[min_area == INF] = self.background_id # set as unannotated
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, ids, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
