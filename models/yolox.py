# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 22:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import time
import numpy as np
import torch
import torch.nn as nn
import os

from models.backbone import CSPDarknet
from models.backbone.swin_transformer import SwinTransformer
from models.neck.yolo_fpn import YOLOXPAFPN
from models.neck.swin_fpn import Swin_FPN
from models.head.yolo_head import YOLOXHead
from models.losses import YOLOXLoss
from models.post_process import yolox_post_process
from models.ops import fuse_model
from data.data_augment import preproc
from utils.model_utils import load_model
from utils.util import sync_time
from torchinfo import summary


def get_model(opt):
    # define backbone
    backbone_cfg = {"nano": [0.33, 0.25],
                    "tiny": [0.33, 0.375],
                    "s": [0.33, 0.5],
                    "m": [0.67, 0.75],
                    "l": [1.0, 1.0],
                    "x": [1.33, 1.25]}
    depth, width = backbone_cfg[opt.backbone.split("-")[1]]  # "CSPDarknet-s
    in_channel = [256, 512, 1024]
    
    #define backbone and neck
    if opt.backbone.split("-")[0] == "Swin":
        in_channel = [192, 384, 768]
        backbone = SwinTransformer(pretrain_img_size=224, yolo_width=width)
        neck = YOLOXPAFPN(depth=depth, width=width, in_channels=in_channel, depthwise=opt.depth_wise)
    else:
        backbone = CSPDarknet(dep_mul=depth, wid_mul=width, out_indices=(3, 4, 5), depthwise=opt.depth_wise)
        neck = YOLOXPAFPN(depth=depth, width=width, in_channels=in_channel, depthwise=opt.depth_wise)
    
    # define head
    head = YOLOXHead(num_classes=opt.num_classes, reid_dim=opt.reid_dim, width=width, in_channels=in_channel,
                     depthwise=opt.depth_wise)
    # define loss
    loss = YOLOXLoss(opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums, strides=opt.stride,
                     in_channels=in_channel)
    # define network
    model = YOLOX(opt, backbone=backbone, neck=neck, head=head, loss=loss)

    model_stats = summary(model)
    """backbone_stats = summary(backbone)
    neck_stats = summary(neck)
    head_stats = summary(head)

    print("Backbone trainable params: " + str(backbone_stats.trainable_params) + " total params: " + str(backbone_stats.total_params))

    print("Head+Neck trainable params: " + str(neck_stats.trainable_params+head_stats.trainable_params) + " total params: " + str(neck_stats.total_params+head_stats.total_params))"""
    
    return model


class YOLOX(nn.Module):
    def __init__(self, opt, backbone, neck, head, loss):
        super(YOLOX, self).__init__()
        self.opt = opt
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

        if opt.swin_pretrained:
            self.load_swin_backbone_weights(opt.swin_weights_path)
            print("Swin-T weights loaded and frozen")
        elif opt.csp_pretrained:
            self.load_csp_backbone_weights(opt.csp_weights_path)
            print("CSP weights loaded and frozen")
        else:
            self.backbone.init_weights()

        if opt.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.neck.init_weights()
        self.head.init_weights()
    
    def load_swin_backbone_weights(self, weights_path):
        swin_dict = self.backbone.state_dict()

        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        coco_pretrain = torch.load(weights_path, map_location=map_location)['state_dict']
        print("Used pretrained model parameters:{}".format(coco_pretrain.keys()))

        swin_dict.update(coco_pretrain)
        self.backbone.load_state_dict(swin_dict, strict=False)

    def load_csp_backbone_weights(self, weights_path):
        assert os.path.isfile(weights_path), "model {} not find".format(weights_path)
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
        print('==>> loaded {}'.format(weights_path))
        state_dict_ = checkpoint['state_dict']


        state_dict_ = {k.replace('backbone.',''): v for k,v in state_dict_.items() if 'backbone' in k}
        model_state_dict = self.backbone.state_dict()

        for k in state_dict_:
            if k in model_state_dict:
                print("Used parameter:{}".format(k))
            else:
                print("Not used parameter:{}".format(k))


        model_state_dict.update(state_dict_)

       

        
        # for index, (k, v) in enumerate(state_dict.items()):
        #    print("Load pretrained weights: {}, {}, {}".format(index, k, v.size()))

        self.backbone.load_state_dict(model_state_dict, strict=False)
    

    def forward(self, inputs, targets=None, show_time=False):
        with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
            if show_time:
                s1 = sync_time(inputs)

            body_feats = self.backbone(inputs)
            neck_feats = self.neck(body_feats)
            yolo_outputs = self.head(neck_feats)
            # print('yolo_outputs:', [[i.shape, i.dtype, i.device] for i in yolo_outputs])  # float16 when use_amp=True

            if show_time:
                s2 = sync_time(inputs)
                print("[inference] batch={} time: {}s".format("x".join([str(i) for i in inputs.shape]), s2 - s1))

            if targets is not None:
                loss = self.loss(yolo_outputs, targets)
                # for k, v in loss.items():
                #     print(k, v, v.dtype, v.device)  # always float32

        if targets is not None:
            return yolo_outputs, loss
        else:
            return yolo_outputs


class Detector(object):
    def __init__(self, cfg):
        self.opt = cfg
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt.pretrained = None
        self.model = get_model(self.opt)
        print("Loading model {}".format(self.opt.load_model))
        self.model = load_model(self.model, self.opt.load_model)
        self.model.to(self.opt.device)
        self.model.eval()
        if "fuse" in self.opt and self.opt.fuse:
            print("==>> fuse model's conv and bn...")
            self.model = fuse_model(self.model)

    def run(self, images, vis_thresh, show_time=False):
        batch_img = True
        if np.ndim(images) == 3:
            images = [images]
            batch_img = False

        with torch.no_grad():
            if show_time:
                s1 = time.time()

            img_ratios, img_shape = [], []
            inp_imgs = np.zeros([len(images), 3, self.opt.test_size[0], self.opt.test_size[1]], dtype=np.float32)
            for b_i, image in enumerate(images):
                img_shape.append(image.shape[:2])
                img, r = preproc(image, self.opt.test_size, self.opt.rgb_means, self.opt.std)
                inp_imgs[b_i] = img
                img_ratios.append(r)

            if show_time:
                s2 = time.time()
                print("[pre_process] time {}".format(s2 - s1))

            inp_imgs = torch.from_numpy(inp_imgs).to(self.opt.device)
            yolo_outputs = self.model(inp_imgs, show_time=show_time)

            if show_time:
                s3 = sync_time(inp_imgs)
            predicts = yolox_post_process(yolo_outputs, self.opt.stride, self.opt.num_classes, vis_thresh,
                                          self.opt.nms_thresh, self.opt.label_name, img_ratios, img_shape)
            if show_time:
                s4 = sync_time(inp_imgs)
                print("[post_process] time {}".format(s4 - s3))
        if batch_img:
            return predicts
        else:
            return predicts[0]
