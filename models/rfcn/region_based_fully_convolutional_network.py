import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import utils.array_tool as at
from utils.config import opt
from models.faster_rcnn_base import LossTuple
from models.utils.bbox_tools import loc2bbox
from models.utils.backbone_loader import load_vgg16_as_fully_convolutional
from models.rpn.region_proposal_network import _RPN


class RFCN(nn.Module):
    def __init__(self, n_fg_class):
        super(RFCN, self).__init__()
        extractor = load_vgg16_as_fully_convolutional(pool_conv5=False)
        self.n_class = n_fg_class + 1
        self.extractor = extractor
        self.rpn = _RPN(
            is_fpn=False,
            in_chs=1024,
            mid_chs=512,
            scales=[128, 256, 512],
            ratios=[0.5, 1, 2],
            n_anchor=9,
            feat_strides=16
        )

        # shape of feature grid -> 7 x 7 (same as pooling size)
        self.loc = nn.Conv2d(1024, 7 * 7 * 4 * self.n_class, 1, 1)
        self.score = nn.Conv2d(1024, 7 * 7 * self.n_class, 1, 1)
        normal_init(self.loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

        self.spatial_scale = 1/16.
        self.pooling_size = 7
        self.roi_sigma = opt.roi_sigma

        # variables for eval mode
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    def forward(self, x, scale, gt_bboxes, gt_labels, original_size=None):
        if self.training:
            img_size = tuple(x.shape[2:])

            # Feature extractor from the base network(e.g. VGG16, ResNet)
            feature = self._extract_features(x)

            # Region Proposal Network
            rpn_result = self.rpn(feature, img_size, scale, gt_bboxes[0], gt_labels[0])
            roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss = rpn_result

            # bbox regression & classification
            roi_loc, roi_score = self._bbox_regression_and_classification(feature, roi)

            # Faster R-CNN loss
            n_sample = roi_loc.shape[0]
            roi_loc = roi_loc.view(n_sample, -1, 4)
            roi_loc = roi_loc[t.arange(0, n_sample).long().cuda(),
                              at.totensor(gt_roi_label).long()]

            gt_roi_loc = at.totensor(gt_roi_loc)
            gt_roi_label = at.totensor(gt_roi_label).long()

            roi_loc_loss = bbox_regression_loss(
                roi_loc.contiguous(),
                gt_roi_loc,
                gt_roi_label.data,
                self.roi_sigma
            )

            roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label.cuda())

            # Stack losses
            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
            losses = losses + [sum(losses)]

            return LossTuple(*losses)
        else:
            with t.no_grad():
                x = at.totensor(x).float()
                img_size = tuple(x.shape[2:])

                # Feature extractor from the base network(e.g. VGG16, ResNet)
                feature = self._extract_features(x)

                # Region Proposal Network
                roi = self.rpn(feature, img_size, scale, None, None)

                # bbox regression & classification
                roi_loc, roi_score = self._bbox_regression_and_classification(feature, roi)

                roi_loc = roi_loc.data
                roi_score = roi_score.data
                roi = at.totensor(roi) / scale

                # Convert predictions to bounding boxes in image coordinates.
                # Bounding boxes are scaled to the scale of the input images.
                mean = t.tensor(self.loc_normalize_mean).cuda(). \
                    repeat(self.n_class)[None]
                std = t.tensor(self.loc_normalize_std).cuda(). \
                    repeat(self.n_class)[None]

                roi_loc = (roi_loc * std + mean)
                roi_loc = roi_loc.view(-1, self.n_class, 4)

                roi = roi.view(-1, 1, 4).expand_as(roi_loc)
                bbox = loc2bbox(at.tonumpy(roi).reshape(-1, 4),
                                at.tonumpy(roi_loc).reshape(-1, 4))
                bbox = at.totensor(bbox)
                bbox = bbox.view(-1, self.n_class * 4)

                # clip bbox
                bbox[:, 0::2] = bbox[:, 0::2].clamp(min=0, max=original_size[0])
                bbox[:, 1::2] = bbox[:, 1::2].clamp(min=0, max=original_size[1])

                prob = F.softmax(at.totensor(roi_score), dim=1)

                bbox, label, score = self._suppress(bbox, prob)

                return bbox, label, score

    def _extract_features(self, x):
        return self.extractor(x)

    def _bbox_regression_and_classification(self, feature, roi):
        pos_loc_maps = self.loc(feature)
        pos_score_maps = self.score(feature)

        roi_loc = self._roi_pool(pos_loc_maps, roi).mean(3).mean(2)
        roi_score = self._roi_pool(pos_score_maps, roi).mean(3).mean(2)

        return roi_loc, roi_score

    def _roi_pool(self, feature, roi):
        index_and_roi = t.cat(
            [t.zeros(roi.shape[0], 1).cuda(), at.totensor(roi).float()],
            dim=1
        )
        # yx -> xy
        index_and_roi = index_and_roi[:, [0, 2, 1, 4, 3]].contiguous()
        # RoI-Pooling
        roi_pool_feat = tv.ops.ps_roi_pool(
            feature,
            index_and_roi,
            self.pooling_size,
            self.spatial_scale
        )

        return roi_pool_feat

    def _suppress(self, raw_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            bbox_l = raw_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            bbox_l = bbox_l[mask]
            prob_l = prob_l[mask]
            keep = tv.ops.nms(bbox_l, prob_l, self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            bbox.append(bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - data].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


def normal_init(layer, mean, std):
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()


def bbox_regression_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()