import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import utils.array_tool as at
from models.faster_rcnn_base import FasterRCNNBase
from models.utils.backbone_loader import load_vgg16
from models.rpn.region_proposal_network import _RPN
from utils.config import opt


class FPN(FasterRCNNBase):
    def __init__(self, n_fg_class):
        extractor, top_layer = load_vgg16(pretrained=True)
        super(FPN, self).__init__(
            n_class=n_fg_class + 1,
            extractor=extractor,
            rpn=_RPN(
                is_fpn=True,
                in_chs=512,
                mid_chs=512,
                scales=[128, 256, 512],
                ratios=[0.5, 1, 2],
                n_anchor=3,
                feat_strides=[4, 8, 16]
            ),
            top_layer=top_layer,
            loc=nn.Linear(4096, (n_fg_class + 1) * 4),
            score=nn.Linear(4096, n_fg_class + 1),
            spatial_scale=[1/4., 1/8., 1/16.],
            pooling_size=7,
            roi_sigma=opt.roi_sigma
        )
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # lateral, top-down layer
        self.lateral1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.lateral2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.lateral3 = nn.Conv2d(256, 512, 1, 1, 0)
        self.smooth1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.smooth2 = nn.Conv2d(512, 512, 3, 1, 1)
        normal_init(self.lateral1, 0, 0.01)
        normal_init(self.lateral2, 0, 0.01)
        normal_init(self.lateral3, 0, 0.01)
        normal_init(self.smooth1, 0, 0.01)
        normal_init(self.smooth2, 0, 0.01)

        self.default_roi_level = 5

    def _extract_features(self, x):
        features = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in [15, 22, 29]:
                features.append(x)

        # lateral & top-down connection
        p5 = self.lateral1(features[2])
        p4 = self._upsample_add(p5, self.lateral2(features[1]))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.lateral3(features[0]))
        p3 = self.smooth2(p3)

        return [p3, p4, p5]

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def _roi_pool(self, feature, roi):
        roi = at.totensor(roi).float()
        h = roi.data[:, 2] - roi.data[:, 0]
        w = roi.data[:, 3] - roi.data[:, 1]
        roi_level = t.log2(t.sqrt(h * w) / 224.)
        roi_level = t.round(roi_level + self.default_roi_level)
        roi_level[roi_level < 3] = 3
        roi_level[roi_level > 5] = 5

        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(3, 6)):
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
                feature[i],
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

        # Classifier from the base network
        fc7 = self.top_layer(roi_pool_feat)

        # bbox regression & classification
        roi_loc = self.loc(fc7)
        roi_score = self.score(fc7)

        return roi_loc, roi_score


def normal_init(layer, mean, std):
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()