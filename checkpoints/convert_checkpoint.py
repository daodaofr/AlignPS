import torch
num_classes = 1
#model_coco = torch.load("fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco_20200603-67b3859f.pth")
#model_coco = torch.load("fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth")  
#model_coco = torch.load("fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20200314-e201886d.pth")
#model_coco = torch.load("fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20200316-a668468b.pth")
model_coco = torch.load("fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20200314-065d37a6.pth")
# weight
#print(model_coco["state_dict"].keys())
print(model_coco["state_dict"]["bbox_head.fcos_cls.weight"].shape) # 80 256 3 3
print(model_coco["state_dict"]["bbox_head.fcos_cls.bias"].shape)
model_coco["state_dict"]["bbox_head.fcos_cls.weight"] = model_coco["state_dict"]["bbox_head.fcos_cls.weight"][:num_classes, :]
model_coco["state_dict"]["bbox_head.fcos_cls.bias"] = model_coco["state_dict"]["bbox_head.fcos_cls.bias"][:num_classes]

#save new model
#torch.save(model_coco,"fcos_coco_pretrained_weights_classes_%d.pth"%num_classes)
#torch.save(model_coco,"fcos_x101_coco_pretrained_weights_classes_%d.pth"%num_classes)
#torch.save(model_coco,"fcos_hrnetv2p_w40_coco_pretrained_weights_classes_%d.pth"%num_classes)
#torch.save(model_coco,"fcos_hrnetv2p_w18_coco_pretrained_weights_classes_%d.pth"%num_classes)
torch.save(model_coco,"fcos_hrnetv2p_w32_coco_pretrained_weights_classes_%d.pth"%num_classes)
