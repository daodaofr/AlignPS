from ..builder import DETECTORS
from .single_stage_reid_fpn import SingleStageReidFpnDetector


@DETECTORS.register_module()
class FCOSReidFpn(SingleStageReidFpnDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSReidFpn, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
