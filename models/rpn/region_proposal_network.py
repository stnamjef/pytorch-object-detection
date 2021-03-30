import torch as t
import torch.nn as nn
import torch.nn.functional as F
import utils.array_tool as at
from utils.config import opt
from models.rpn.proposal_layer import _ProposalLayer
from models.rpn.proposal_target_layer import _ProposalTargetLayer
from models.rpn.anchor_target_layer import _AnchorTargetLayer
from models.utils.bbox_tools import generate_anchors, generate_anchors_fpn


class _RPN(nn.Module):
    def __init__(self, is_fpn, in_chs, mid_chs, scales, ratios, n_anchor, feat_strides):
        super(_RPN, self).__init__()
        self.is_fpn = is_fpn
        self.scales = scales
        self.ratios = ratios
        self.n_anchor = n_anchor
        self.feat_strides = feat_strides
        self.proposal_layer = _ProposalLayer(
            self,
            nms_thresh=opt.NMS_THRESH,
            n_train_pre_nms=opt.N_TRAIN_PRE_NMS,
            n_train_post_nms=opt.N_TRAIN_POST_NMS,
            n_test_pre_nms=opt.N_TEST_PRE_NMS,
            n_test_post_nms=opt.N_TEST_POST_NMS,
            min_size=opt.MIN_SIZE
        )
        self.proposal_target_layer = _ProposalTargetLayer(
            n_sample=opt.N_SAMPLE_PROPOSAL,
            pos_ratio=opt.POS_RATIO_PROPOSAL,
            pos_iou_thresh=opt.POS_IOU_THRESH_PROPOSAL,
            neg_iou_thresh_hi=opt.NEG_IOU_THRESH_HI_PROPOSAL,
            neg_iou_thresh_lo=opt.NEG_IOU_THRESH_LO_PROPOSAL
        )
        self.anchor_target_layer = _AnchorTargetLayer(
            n_sample=opt.N_SAMPLE_ANCHOR,
            pos_iou_thresh=opt.POS_IOU_THRESH_ANCHOR,
            neg_iou_thresh=opt.NEG_IOU_THRESH_ANCHOR,
            pos_ratio=opt.POS_RATIO_ANCHOR
        )
        self.rpn_conv = nn.Conv2d(in_chs, mid_chs, 3, 1, 1)
        self.rpn_loc = nn.Conv2d(mid_chs, n_anchor * 4, 1, 1, 0)
        self.rpn_score = nn.Conv2d(mid_chs, n_anchor * 2, 1, 1, 0)
        normal_init(self.rpn_conv, 0, 0.01)
        normal_init(self.rpn_loc, 0, 0.01)
        normal_init(self.rpn_score, 0, 0.01)

        # mean and std
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

        self.rpn_sigma = opt.rpn_sigma

    def forward(self, features, img_size, scale, gt_bbox, gt_label):
        if self.is_fpn:
            feat_shapes = [tuple(f.shape[2:]) for f in features]
            anchor = generate_anchors_fpn(self.scales, self.ratios, feat_shapes, self.feat_strides)
        else:
            features = [features]
            feat_shapes = [tuple(f.shape[2:]) for f in features]
            anchor = generate_anchors(self.scales, self.ratios, feat_shapes[0], self.feat_strides)

        n = 1  # batch size is always one
        locs = []
        scores = []
        fg_scores = []
        for x, (h, w) in zip(features, feat_shapes):
            x2 = F.relu(self.rpn_conv(x))

            loc = self.rpn_loc(x2)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

            score = self.rpn_score(x2)
            score = score.permute(0, 2, 3, 1).contiguous()

            softmax_score = F.softmax(score.view(n, h, w, self.n_anchor, 2), dim=4)

            fg_score = softmax_score[:, :, :, :, 1].contiguous().view(n, -1)

            locs.append(loc)
            scores.append(score.view(n, -1, 2))
            fg_scores.append(fg_score)

        loc = t.cat(locs, dim=1)[0]
        score = t.cat(scores, dim=1)[0]
        fg_score = t.cat(fg_scores, dim=1)[0]

        roi = self.proposal_layer(
            loc.cpu().data.numpy(),
            fg_score.cpu().data.numpy(),
            anchor,
            img_size,
            scale
        )

        if self.training:
            # if training phase, then sample RoIs
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(gt_label),
                self.loc_normalize_mean,
                self.loc_normalize_std
            )

            # get gt_loc(offset from anchor to gt_bbox)
            gt_rpn_loc, gt_rpn_label = self.anchor_target_layer(
                at.tonumpy(gt_bbox),
                anchor,
                img_size
            )
            gt_rpn_loc = at.totensor(gt_rpn_loc)
            gt_rpn_label = at.totensor(gt_rpn_label).long()

            # bounding-box regression loss
            rpn_loc_loss = bbox_regression_loss(
                loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma
            )

            # foreground-background classification loss
            rpn_cls_loss = F.cross_entropy(score, gt_rpn_label.cuda(), ignore_index=-1)

            return sample_roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss

        return roi


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