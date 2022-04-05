import torch
import torch.nn as nn
import torchvision

from .network_blocks import BaseConv



class Swin_FPN(nn.Module):


    def __init__(
        self,
        yolo_width = 1.0,
    ):
        super().__init__()
        
        self.fpn_top_down = torchvision.ops.FeaturePyramidNetwork([96, 192, 384, 768], 256)
        self.fpn_bot_up = torchvision.ops.FeaturePyramidNetwork([256, 256, 256, 256], 256)
        

        self.cbl1 = BaseConv(256, int(yolo_width*64*4), 1, stride=1, act="lrelu")
        self.cbl2 = BaseConv(256, int(yolo_width*64*8), 1, stride=1, act="lrelu")
        self.cbl3 = BaseConv(256, int(yolo_width*64*16), 1, stride=1, act="lrelu")
        


        """# out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)
        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)
        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")"""


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


    def forward(self, inputs):

        
        
        #ordered dict that works with torchvision fpn
        #increasing depth(channel number), decreasing resolution
        features = {
            "x0": inputs[0],
            "x1": inputs[1],
            "x2": inputs[2],
            "x3": inputs[3],
        }

        
        #top-down fpn pathway
        #channelSize=256, decreasing resolution
        fpn_top_down = self.fpn_top_down(features)


        #reversing output of top-down pathway
        features = {
            "x0": fpn_top_down["x3"],
            "x1": fpn_top_down["x2"],
            "x2": fpn_top_down["x1"],
            "x3": fpn_top_down["x0"],
        }

        #bot-up fpn pathway
        fpn_bot_up = self.fpn_bot_up(features)

        #reverse again
        features = {
            "x0": fpn_bot_up["x3"],
            "x1": fpn_bot_up["x2"],
            "x2": fpn_bot_up["x1"],
            "x3": fpn_bot_up["x0"],
        }


        #change dimensions to head input requirements
        features["x1"] = self.cbl1(features["x1"])
        features["x2"] = self.cbl2(features["x2"])
        features["x3"] = self.cbl3(features["x3"])

        #print(str(fpn_top_down["x2"].size()) + str(fpn_top_down["x1"].size()) + str(fpn_top_down["x0"].size()))
        #print(str(fpn_bot_up["x2"].size()) + str(fpn_bot_up["x1"].size()) + str(fpn_bot_up["x0"].size()))
        #print(str(features["x0"].size()) + str(features["x1"].size()) + str(features["x2"].size()) + str(features["x3"].size()))

        #x2 = 128, x1 = 256, x0 = 512
        #high res -> low res
        #small dimension -> high dimension

        outputs = (features["x1"], features["x2"], features["x3"])

        
        return outputs

if __name__ == "__main__":

    in_channels = [96, 192, 384, 768]
    feats = [torch.rand([1, in_channels[0], 64, 64]), torch.rand([1, in_channels[1], 32, 32]),
             torch.rand([1, in_channels[2], 16, 16]), torch.rand([1, in_channels[3], 4, 4])]

    # fpn = PPYOLOPAN(in_channels, norm_type='bn', act='mish', conv_block_num=3, drop_block=True, block_size=3, spp=True)
    # fpn = PPYOLOFPN(in_channels, coord_conv=True, drop_block=True, block_size=3, keep_prob=0.9, spp=True)
    # fpn = YOLOv3FPN(in_channels)
    # fpn = PPYOLOTinyFPN(in_channels)
    fpn = Swin_FPN()
    fpn.init_weights()
    # print(fpn)
    fpn.eval()
    # total_ops, total_params = profile(fpn, (feats,))
    # print("total_ops {:.2f}G, total_params {:.2f}M".format(total_ops/1e9, total_params/1e6))
    output = fpn(feats)
    for o in output:
        print(o.size())