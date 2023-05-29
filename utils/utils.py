import torch.nn as nn
import torch

class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()

class SelfTrainingLoss(nn.Module):
    requires_reduction = False

    def __init__(self, conf_th=0.9, fraction=None, ignore_index=255, lambda_selftrain=1, **kwargs):
        super().__init__()
        self.conf_th = conf_th
        self.fraction = fraction
        self.ignore_index = ignore_index
        self.teacher = None
        self.lambda_selftrain = lambda_selftrain

    def set_teacher(self, model):
        self.teacher = model

    def get_image_mask(self, prob, pseudo_lab):
        max_prob = prob.detach().clone().max(0)[0]
        mask_prob = max_prob > self.conf_th if 0. < self.conf_th < 1. else torch.zeros(max_prob.size(),
                                                                                       dtype=torch.bool).to(
            max_prob.device)
        if self.fraction is not None:
            mask_topk = torch.zeros(max_prob.size(), dtype=torch.bool).to(max_prob.device)
            if 0. < self.fraction < 1.:
                for c in pseudo_lab.unique():
                    mask_c = pseudo_lab == c
                    max_prob_c = max_prob.clone()
                    max_prob_c[~mask_c] = 0
                    _, idx_c = torch.topk(max_prob_c.flatten(), k=int(mask_c.sum() * self.fraction))
                    mask_topk_c = torch.zeros_like(max_prob_c.flatten(), dtype=torch.bool)
                    mask_topk_c[idx_c] = 1
                    mask_c &= mask_topk_c.unflatten(dim=0, sizes=max_prob_c.size())
                    mask_topk |= mask_c
            return mask_prob | mask_topk
        else:
            return mask_prob

    def get_batch_mask(self, pred, pseudo_lab):
        b, _, _, _ = pred.size()
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(F.softmax(pred, dim=1), pseudo_lab)], dim=0)
        return mask

    def get_pseudo_lab(self, pred, imgs=None, return_mask_fract=False, model=None):
        teacher = self.teacher if model is None else model
        if teacher is not None:
            with torch.no_grad():
                try:
                    pred = teacher(imgs)['out']
                except:
                    pred = teacher(imgs)
                pseudo_lab = pred.detach().max(1)[1]
        else:
            pseudo_lab = pred.detach().max(1)[1]
        mask = self.get_batch_mask(pred, pseudo_lab)
        pseudo_lab[~mask] = self.ignore_index
        if return_mask_fract:
            return pseudo_lab, F.softmax(pred, dim=1), mask.sum() / mask.numel()
        return pseudo_lab

    def forward(self, pred, imgs=None):
        pseudo_lab = self.get_pseudo_lab(pred, imgs)
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean() * self.lambda_selftrain