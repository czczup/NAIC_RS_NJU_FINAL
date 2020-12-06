"""Custom losses."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .lovasz_losses import lovasz_softmax
from ..models.pointrend import point_sample
from ..data.dataloader import datasets
from ..config import cfg

__all__ = ['get_segmentation_loss']


class LabelSmoothingCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(LabelSmoothingCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def _aux_forward(self, preds, target, masked_indices, **kwargs):
        loss = self.cross_entropy(preds[0], target, masked_indices)
        for i in range(1, len(preds)):
            aux_loss = self.cross_entropy(preds[i], target, masked_indices)
            loss += self.aux_weight * aux_loss
        return loss

    def _label_smoothing(self, target: torch.Tensor, classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        label_shape = torch.Size((target.size(0), classes,
                                  target.size(1), target.size(2)))
        with torch.no_grad():
            target_smooth = torch.empty(size=label_shape, device=target.device)
            target_smooth.fill_(smoothing / (classes - 1))
            target_smooth.scatter_(dim=1, index=target.data.unsqueeze(1),
                                   value=confidence)
        return target_smooth

    def cross_entropy(self, pred, target, masked_indices):
        pred = F.log_softmax(pred, dim=1)
        loss = -(target * pred).sum(1)
        loss.masked_fill_(masked_indices, 0)
        loss = loss.sum() / float(loss.size(0)*loss.size(1)*loss.size(2) - masked_indices.sum())
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        num_class = preds[0].size(1)
        masked_indices = target.eq(self.ignore_index)

        target.masked_fill_(masked_indices, 0)

        smoothing_target = self._label_smoothing(target, classes=num_class, smoothing=0.1)

        if self.aux:
            return dict(loss=self._aux_forward(preds, smoothing_target, masked_indices))
        else:
            return dict(loss=self.cross_entropy(preds[0], smoothing_target, masked_indices))



class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _attn_scale_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[1], target)
        loss += self.aux_weight * aux_loss
        for i in range(2, len(preds)):
            attn_scale_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += cfg.TRAIN.SUPERVISED_MSCALE_WT * attn_scale_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss
    
    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            if cfg.TRAIN.SUPERVISED_MSCALE_WT != 0:
                return dict(loss=self._attn_scale_forward(*inputs))
            else:
                return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


class MixSoftmaxCrossEntropyLossV2(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLossV2, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def _aux_forward(self, preds, target):
        loss = F.cross_entropy(preds[0], target, weight=None, ignore_index=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = F.cross_entropy(preds[i], target, weight=None, ignore_index=self.ignore_index)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, targets = tuple(inputs)
        loss_8 = self._aux_forward(preds[0], targets[0])
        loss_14 = self._aux_forward(preds[1], targets[1])
        return dict(loss_8=loss_8*0.5, loss_14=loss_14*0.5)


class MixDistillLossAndCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixDistillLossAndCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.alpha = cfg.DISTILL.ALPHA
        self.temperature = cfg.DISTILL.TEMPERATURE
        logging.info("distill alpha: %.2f, temperature: %d" % (self.alpha, self.temperature))

    def _aux_forward(self, preds, target):
        loss = F.cross_entropy(preds[0], target, weight=None, ignore_index=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = F.cross_entropy(preds[i], target, weight=None, ignore_index=self.ignore_index)
            loss += self.aux_weight * aux_loss
        return loss
    
    def pixel_wise_loss(self, preds, logits, masked_indices, eps=1E-6):
        B, C, W, H = preds.shape
        masked_indices = masked_indices.view(B, -1)  # [8, 640*640]
        soft_log_out = F.log_softmax(preds / self.temperature, dim=1)
        soft_t = logits
        loss_kd = F.kl_div(soft_log_out, soft_t.detach(), reduction="none")  # [8, 8, 640, 640]
        loss_kd = loss_kd.sum(dim=1)  # [8, 640, 640]
        loss_kd = loss_kd.view(B, W * H)  # [8, 640*640]
        loss_kd.masked_fill_(masked_indices, 0)  # [8, 640*640]
        loss = loss_kd.sum(1) / (loss_kd.size(1) - masked_indices.sum(1) + eps)
        loss = loss.mean()
        return loss
    
    def _distill_forward(self, preds, logits, targets):
        assert preds[0].shape == logits.shape, 'the output dim of teacher and student differ'
        masked_indices = targets.eq(self.ignore_index)
        loss = self.pixel_wise_loss(preds[0], logits, masked_indices)
        
        for i in range(1, len(preds)):
            aux_loss = self.pixel_wise_loss(preds[i], logits, masked_indices)
            loss += self.aux_weight * aux_loss
        return loss
    
    def forward(self, *inputs, **kwargs):
        preds, targets, logits = tuple(inputs)
        ce_loss_8 = self._aux_forward(preds[0], targets[0])
        ce_loss_14 = self._aux_forward(preds[1], targets[1])
        distill_loss_8 = self._distill_forward(preds[0], logits[0], targets[0])
        distill_loss_14 = self._distill_forward(preds[1], logits[1], targets[0])
        ce_loss = ce_loss_8 * 0.5 + ce_loss_14 * 0.5
        distill_loss = distill_loss_8 * 0.5 + distill_loss_14 * 0.5
        return dict(ce_loss=ce_loss, distill_loss=distill_loss * self.alpha)


class MixDistillLossAndCrossEntropyLossV2(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixDistillLossAndCrossEntropyLossV2, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.alpha = cfg.DISTILL.ALPHA
        self.beta = cfg.DISTILL.BETA
        self.temperature = cfg.DISTILL.TEMPERATURE
        self.scale = cfg.DISTILL.SCALE
        logging.info("alpha: %d, beta: %f, scale: %f, temperature: %d"
                     % (self.alpha, self.beta, self.scale, self.temperature))
    
    def _aux_forward(self, preds, target):
        loss = F.cross_entropy(preds[0], target, weight=None, ignore_index=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = F.cross_entropy(preds[i], target, weight=None, ignore_index=self.ignore_index)
            loss += self.aux_weight * aux_loss
        return loss
    
    # def pixel_wise_loss(self, pred_s, pred_t):
    #     B, C, W, H = pred_s.shape
    #     soft_log_out = F.log_softmax(pred_s / self.temperature, dim=1)
    #     soft_t = pred_t / self.temperature
    #     loss_kd = F.kl_div(soft_log_out, soft_t.detach(), reduction="none")
    #     loss_kd = loss_kd.sum(dim=1)
    #     loss_kd = loss_kd.view(B, W * H)
    #     loss = loss_kd.sum(1) / loss_kd.size(1)
    #     loss = loss.mean()
    #     return loss
    
    def pixel_wise_loss(self, pred_s, pred_t, masked_indices, eps=1E-6):
        B, C, W, H = pred_s.shape
        masked_indices = masked_indices.view(B, -1)  # [8, 640*640]
        soft_log_out = F.log_softmax(pred_s / self.temperature, dim=1)
        soft_t = F.softmax(pred_t / self.temperature, dim=1)
        loss_kd = F.kl_div(soft_log_out, soft_t.detach(), reduction="none")  # [8, 8, 640, 640]
        loss_kd = loss_kd.sum(dim=1)  # [8, 640, 640]
        loss_kd = loss_kd.view(B, W * H)  # [8, 640*640]
        loss_kd.masked_fill_(masked_indices, 0)  # [8, 640*640]
        loss = loss_kd.sum(1) / (loss_kd.size(1) - masked_indices.sum(1) + eps)
        loss = loss.mean()
        return loss
    
    def _pi_forward(self, preds_s, preds_t, targets):
        assert preds_s[0].shape == preds_t[0].shape, 'the output dim of teacher and student differ'
        masked_indices = targets.eq(self.ignore_index)
        loss = self.pixel_wise_loss(preds_s[0], preds_t[0], masked_indices)
        for i in range(1, len(preds_s)):
            aux_loss = self.pixel_wise_loss(preds_s[i], preds_t[0], masked_indices)
            loss += self.aux_weight * aux_loss
        return loss

    def L2(self, f_):
        return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8

    def similarity(self, feat):
        feat = feat.float()
        tmp = self.L2(feat).detach()
        feat = feat / tmp
        feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
        return torch.einsum('icm,icn->imn', [feat, feat])

    def sim_dis_compute(self, f_S, f_T):
        sim_err = ((self.similarity(f_T) - self.similarity(f_S)) ** 2) / ((f_T.shape[-1] * f_T.shape[-2]) ** 2) / f_T.shape[0]
        sim_dis = sim_err.sum()
        return sim_dis

    def _pa_forward(self, aspp_s, aspp_t):
        assert aspp_s.shape == aspp_t.shape, 'the output dim of teacher and student differ'
        total_w, total_h = aspp_t.shape[2], aspp_t.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)
        loss = self.sim_dis_compute(maxpool(aspp_s), maxpool(aspp_t))
        return loss
    
    def forward(self, *inputs, **kwargs):
        preds_s, preds_t, targets = tuple(inputs)
        ce_loss_8 = self._aux_forward(preds_s[0], targets[0])
        ce_loss_14 = self._aux_forward(preds_s[1], targets[1])
        ce_loss = ce_loss_8 * 0.5 + ce_loss_14 * 0.5

        pi_loss_8 = self._pi_forward(preds_s[0], preds_t[0], targets[0])
        pi_loss_14 = self._pi_forward(preds_s[1], preds_t[1], targets[1])
        pi_loss = pi_loss_8 * 0.5 + pi_loss_14 * 0.5

        pa_loss_8 = self._pa_forward(preds_s[2][0], preds_t[2][0])
        pa_loss_14 = self._pa_forward(preds_s[2][1], preds_t[2][1])
        pa_loss = pa_loss_8 * 0.5 + pa_loss_14 * 0.5
        
        return dict(ce_loss=ce_loss, pi_loss=pi_loss * self.alpha,
                    pa_loss=pa_loss * self.beta)

class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    def __init__(self, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def _forward(self, preds, target):
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight
    
    def forward(self, *inputs, **kwargs):
        preds, targets = tuple(inputs)
        loss_8 = self._forward(preds[0], targets[0])
        loss_14 = self._forward(preds[1], targets[1])
        return dict(loss_8=loss_8*0.5, loss_14=loss_14*0.5)

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
            # weight = torch.FloatTensor([0.856, 0.486, 1.364, 1.296,
            #                             0.688, 1.205, 0.409, 1.373])

            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            # prob = prob.masked_fill_(1 - valid_mask, 1)
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        # target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = cfg.MODEL.ENCNET.SE_LOSS
        self.se_weight = cfg.MODEL.ENCNET.SE_WEIGHT
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _attn_scale_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[1], target)
        loss += self.aux_weight * aux_loss
        for i in range(2, len(preds)):
            attn_scale_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += cfg.TRAIN.SUPERVISED_MSCALE_WT * attn_scale_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            if cfg.TRAIN.SUPERVISED_MSCALE_WT != 0:
                logging.info("use supervised mscale weight!")
                return dict(loss=self._attn_scale_forward(*inputs))
            else:
                return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))


class MixLovaszSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(MixLovaszSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)

        loss_lovasz = lovasz_softmax(F.softmax(preds[0], dim=1), target, ignore=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = lovasz_softmax(F.softmax(preds[i], dim=1), target, ignore=self.ignore_index)
            loss_lovasz += self.aux_weight * aux_loss

        loss_ce = super(MixLovaszSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixLovaszSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss_ce += self.aux_weight * aux_loss
        return dict(loss=0.5*loss_lovasz+0.5*loss_ce)


class MixLovaszSoftmaxCrossEntropyLossV2(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(MixLovaszSoftmaxCrossEntropyLossV2, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def _aux_forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)

        loss_lovasz = lovasz_softmax(F.softmax(preds[0], dim=1), target, ignore=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = lovasz_softmax(F.softmax(preds[i], dim=1), target, ignore=self.ignore_index)
            loss_lovasz += self.aux_weight * aux_loss

        loss_ce = super(MixLovaszSoftmaxCrossEntropyLossV2, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixLovaszSoftmaxCrossEntropyLossV2, self).forward(preds[i], target)
            loss_ce += self.aux_weight * aux_loss
        return 0.5*loss_lovasz+0.5*loss_ce

    def forward(self, *inputs, **kwargs):
        preds, targets = tuple(inputs)
        loss_8 = self._aux_forward(preds[0], targets[0])
        loss_14 = self._aux_forward(preds[1], targets[1])
        return dict(loss_8=loss_8*0.5, loss_14=loss_14*0.5)

class LovaszSoftmax(nn.Module):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        loss = lovasz_softmax(F.softmax(preds[0], dim=1), target, ignore=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = lovasz_softmax(F.softmax(preds[i], dim=1), target, ignore=self.ignore_index)
            loss += self.aux_weight * aux_loss
        return dict(loss=loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, aux=True, aux_weight=0.2, ignore_index=-1,
                 size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = self._base_forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _base_forward(self, output, target):

        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).long()
        target_one_hot = F.one_hot(torch.clamp_min(target, 0))
        loss = self._base_forward(preds[0], target_one_hot, valid_mask)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


class PointRendLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(PointRendLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def forward(self, *inputs, **kwargs):
        result, gt = tuple(inputs)
        
        pred = F.interpolate(result["coarse"], gt.shape[-2:], mode="bilinear", align_corners=True)
        seg_loss = F.cross_entropy(pred, gt, ignore_index=self.ignore_index)

        gt_points = point_sample(
            gt.float().unsqueeze(1),
            result["points"],
            mode="nearest",
            align_corners=False
        ).squeeze_(1).long()
        points_loss = F.cross_entropy(result["rend"], gt_points, ignore_index=self.ignore_index)

        loss = seg_loss + points_loss

        return dict(loss=loss)


class MixRMILoss(nn.Module):
    def __init__(self, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(MixRMILoss, self).__init__()
        from .rmi import RMILoss
        self.RMI_loss = RMILoss(num_classes=8, ignore_index=8)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = self.RMI_loss.forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = self.RMI_loss.forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=self.RMI_loss.forward(*inputs))


def get_segmentation_loss(model, use_ohem=False, **kwargs):
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'lovasz':
        logging.info('Use lovasz loss!')
        return LovaszSoftmax(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'lovasz-ce':
        logging.info('Use lovasz loss and cross entropy!')
        return MixLovaszSoftmaxCrossEntropyLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'lovasz-ce-v2':
        logging.info('Use lovasz loss and cross entropy!')
        return MixLovaszSoftmaxCrossEntropyLossV2(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'focal':
        logging.info('Use focal loss!')
        return FocalLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'dice':
        logging.info('Use dice loss!')
        return DiceLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'label-smoothing':
        logging.info('Use label smoothing loss!')
        return LabelSmoothingCrossEntropyLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'rmi':
        logging.info('Use RMI loss!')
        return MixRMILoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'distill':
        logging.info('Use distill loss!')
        return MixDistillLossAndCrossEntropyLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'distillv2':
        logging.info('Use distillv2 loss!')
        return MixDistillLossAndCrossEntropyLossV2(**kwargs)
    model = model.lower()
    if model == 'icnet':
        return ICNetLoss(**kwargs)
    elif model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'pointrend':
        logging.info('Use pointrend loss!')
        return PointRendLoss(**kwargs)
    elif model == 'bisenet':
        logging.info('Use two-dataset loss!')
        return MixSoftmaxCrossEntropyLossV2(**kwargs)
    elif 'deeplabv3_plus_v' in model or "dunetv" in model:
        logging.info('Use two-dataset loss!')
        return MixSoftmaxCrossEntropyLossV2(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)




