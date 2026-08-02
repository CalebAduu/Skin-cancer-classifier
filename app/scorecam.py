"""
Score-CAM for the hybrid CNN-ViT, extracted from the thesis notebook.

Targets the last block of ResNet stage 2 in the hybrid stem
(`model.vit.patch_embed.backbone.stages[2].blocks[-1]`), exactly as in the
notebook. For the champion (`use_derm_token=False`) the Derm embedding is not
needed, so `x_derm` defaults to None.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ScoreCAM:
    def __init__(self, model, batch_size=16):
        self.model = model
        self.batch_size = batch_size
        self.activations = None
        target = model.vit.patch_embed.backbone.stages[2].blocks[-1]
        self._handle = target.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output.detach()

    def remove(self):
        self._handle.remove()

    @torch.no_grad()
    def generate_cam(self, x_img, x_num, x_cat, x_derm=None, target_class=None):
        self.model.eval()
        logits = self.model(x_img, x_num, x_cat, x_derm)
        probs = F.softmax(logits, dim=1)
        pred_prob, pred_class = probs.max(1)
        pred_class = pred_class.item()
        pred_prob = pred_prob.item()
        if target_class is None:
            target_class = pred_class

        acts = self.activations[0]
        n_channels, H, W = acts.shape
        channel_scores = []

        for start in range(0, n_channels, self.batch_size):
            end = min(start + self.batch_size, n_channels)
            batch_acts = acts[start:end]
            B_ch = batch_acts.shape[0]

            masks = batch_acts.unsqueeze(1).float()
            masks = F.interpolate(masks, size=(224, 224), mode="bilinear", align_corners=False)
            masks_flat = masks.view(B_ch, -1)
            m_min = masks_flat.min(dim=1).values.view(B_ch, 1, 1, 1)
            m_max = masks_flat.max(dim=1).values.view(B_ch, 1, 1, 1)
            masks = (masks - m_min) / (m_max - m_min + 1e-8)

            masked_imgs = x_img.expand(B_ch, -1, -1, -1) * masks
            num_e = x_num.expand(B_ch, -1)
            cat_e = x_cat.expand(B_ch, -1)
            derm_e = x_derm.expand(B_ch, -1) if x_derm is not None else None

            masked_logits = self.model(masked_imgs, num_e, cat_e, derm_e)
            masked_probs = F.softmax(masked_logits, dim=1)
            channel_scores.extend(masked_probs[:, target_class].cpu().numpy().tolist())

        channel_scores = np.array(channel_scores)
        weights = np.exp(channel_scores - channel_scores.max())
        weights = weights / weights.sum()

        acts_up = F.interpolate(acts.unsqueeze(0).float(), size=(224, 224),
                                mode="bilinear", align_corners=False).squeeze(0)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        cam = (weights_t[:, None, None] * acts_up.cpu()).sum(0).numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, pred_class, pred_prob
