import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import utils.array_tool as at
from models.faster_rcnn_base import FasterRCNNBase
from models.utils.backbone_loader import load_vgg16_as_fully_convolutional
from models.rpn.region_proposal_network import _RPN
from utils.config import opt


class FPNLight(FasterRCNNBase):
    def __init__(self, n_fg_class):
        extractor = load_vgg16_as_fully_convolutional(pool_conv5=True)
        super(FPNLight, self).__init__(
            n_class=n_fg_class + 1,
            extractor=extractor,
            rpn=_RPN(
                is_fpn=True,
                in_chs=256,
                mid_chs=512,
                scales=[64, 128, 256, 512],
                ratios=[0.5, 1, 2],
                n_anchor=3,
                feat_strides=[4, 8, 16, 32]
            ),
            top_layer=nn.Sequential(
                nn.Linear(7 * 7 * 256, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True)
            ),
            loc=nn.Linear(1024, (n_fg_class + 1) * 4),
            score=nn.Linear(1024, n_fg_class + 1),
            spatial_scale=[1/4., 1/8., 1/16., 1/32.],
            pooling_size=7,
            roi_sigma=opt.roi_sigma
        )
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # lateral, top-down layer
        self.lateral1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.lateral2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral3 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral4 = nn.Conv2d(256, 256, 1, 1, 0)
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        normal_init(self.lateral1, 0, 0.01)
        normal_init(self.lateral2, 0, 0.01)
        normal_init(self.lateral3, 0, 0.01)
        normal_init(self.lateral4, 0, 0.01)
        normal_init(self.smooth1, 0, 0.01)
        normal_init(self.smooth2, 0, 0.01)
        normal_init(self.smooth3, 0, 0.01)

        self.default_roi_level = 5

    def _extract_features(self, x):
        features = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in [15, 22, 29, 34]:
                features.append(x)

        # lateral & top-down connection
        p6 = self.lateral1(features[3])
        p5 = self._upsample_add(p6, self.lateral2(features[2]))
        p5 = self.smooth1(p5)
        p4 = self._upsample_add(p5, self.lateral3(features[1]))
        p4 = self.smooth2(p4)
        p3 = self._upsample_add(p4, self.lateral4(features[0]))
        p3 = self.smooth3(p3)

        return [p3, p4, p5, p6]

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def _roi_pool(self, features, roi):
        roi = at.totensor(roi).float()
        h = roi.data[:, 2] - roi.data[:, 0]
        w = roi.data[:, 3] - roi.data[:, 1]
        roi_level = t.sqrt(h * w)
        roi_level[roi_level < 96] = 3
        roi_level[(roi_level >= 96) & (roi_level < 192)] = 4
        roi_level[(roi_level >= 192) & (roi_level < 384)] = 5
        roi_level[roi_level >= 384] = 6

        # roi = at.totensor(roi).float()
        # h = roi.data[:, 2] - roi.data[:, 0]
        # w = roi.data[:, 3] - roi.data[:, 1]
        # roi_level = t.log2(t.sqrt(h * w) / 224.)
        # roi_level = t.round(roi_level + self.default_roi_level)
        # roi_level[roi_level < 3] = 3
        # roi_level[roi_level > 6] = 6

        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(3, 7)):
            if (roi_level == l).sum() == 0:
                continue

            level_idx = t.where(roi_level == l)[0]
            box_to_levels.append(level_idx)

            index_and_roi = t.cat(
                [t.zeros(level_idx.size(0), 1).cuda(), roi[level_idx]],
                dim=1
            )
            # yx -> xy
            index_and_roi = index_and_roi[:, [0, 2, 1, 4, 3]].contiguous()

            feat = tv.ops.roi_pool(
                features[i],
                index_and_roi,
                self.pooling_size,
                self.spatial_scale[i]
            )
            roi_pool_feats.append(feat)

        roi_pool_feat = t.cat(roi_pool_feats, dim=0)
        box_to_level = t.cat(box_to_levels, dim=0)
        idx_sorted, order = t.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]

        return roi_pool_feat

    def _bbox_regression_and_classification(self, roi_pool_feat):
        # flatten roi pooled feature
        roi_pool_feat = roi_pool_feat.view(roi_pool_feat.shape[0], -1)

        # RCNN top_layer
        fc9 = self.top_layer(roi_pool_feat)

        # bbox regression & classification
        roi_loc = self.loc(fc9)
        roi_score = self.score(fc9)

        return roi_loc, roi_score


def normal_init(layer, mean, std):
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()