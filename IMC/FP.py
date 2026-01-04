#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.optim as optim
from attack.add_trigger import add_trigger

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _tanh_func(x: torch.Tensor) -> torch.Tensor:
    """Map real-valued tensor to (0,1) range, matching trojanzoo's tanh_func."""
    return 0.5 * (torch.tanh(x) + 1.0)

def _overlay_patch(images: torch.Tensor, patch: torch.Tensor, y: int, x: int, alpha: float = 1.0):
    """
    Differentiable overlay of patch onto images at (y, x) with optional alpha blend.
    Uses F.pad to avoid in-place writes that break gradients.
    """
    ps = patch.shape[-1]
    # expand patch to batch dimension
    if patch.dim() == 3:  # (C, H, W)
        patch_exp = patch.unsqueeze(0)  # (1, C, H, W)
    else:
        patch_exp = patch
    left = x
    right = images.shape[-1] - x - ps
    top = y
    bottom = images.shape[-2] - y - ps
    padded_patch = F.pad(patch_exp, (left, right, top, bottom))
    # build a mask for blending
    ones_patch = torch.ones_like(patch_exp)
    mask = F.pad(ones_patch, (left, right, top, bottom))
    return images * (1 - alpha * mask) + padded_patch * alpha


class TrojanNNAttack:

    def __init__(self, model, device, args, patch_size=5, neuron_num=2, target_value=100.0,
                 neuron_lr=0.1, neuron_steps=1000, dataset='cifar',
                 preprocess_layer=None, preprocess_next_layer=None,
                 patch_param=None, neuron_idx=None):
        self.model = model
        self.model.eval()
        self.device = device
        self.args = args
        self.patch_size = patch_size

        self.preprocess_layer = preprocess_layer
        self.preprocess_next_layer = preprocess_next_layer


        in_channels = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                in_channels = m.in_channels
                break
        if in_channels is None:
            in_channels = 3 if dataset in ['cifar', 'gtsrb'] else 1
        height = 32 if in_channels > 1 else 28


        init_shape = (in_channels, patch_size, patch_size)
        self.background = torch.zeros((1, in_channels, height, height), device=device)

        if dataset == 'gtsrb' and in_channels == 3:
            mean = torch.tensor([0.3403, 0.3121, 0.3214], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.2724, 0.2608, 0.2669], device=device).view(1, 3, 1, 1)
            self.background -= (mean / std)
        if patch_param is None:
            self.patch_param = torch.randn(init_shape, device=device, requires_grad=True)
        else:
            self.patch_param = patch_param.detach().clone().to(device)
            self.patch_param.requires_grad_(True)
        self.patch_tensor = _tanh_func(self.patch_param.detach())  

        self.target_class = args.attack_label
        self.neuron_num = neuron_num
        # lower default target for gtsrb to avoid instability
        if self.args.dataset == 'gtsrb' and target_value is None:
            self.target_value = 10.0
        else:
            self.target_value = target_value
        self.neuron_lr = neuron_lr
        self.neuron_steps = neuron_steps

        self._extracted_features = None
        self.neuron_idx = neuron_idx  

        if args.dataset == "cifar":

            if self.preprocess_layer is None:
                def hook_fn(module, inp, out):
                    self._extracted_features = out.view(out.size(0), -1)
                self.model.avgpool.register_forward_hook(hook_fn)
            else:
                def hook_fn(module, inp, out):
                    self._extracted_features = out.view(out.size(0), -1)
                self.model.avgpool.register_forward_hook(hook_fn)
            if self.preprocess_next_layer:
                next_name = self.preprocess_next_layer
            else:
                if hasattr(self.model, "linear"):
                    next_name = "linear"
                elif hasattr(self.model, "fc"):
                    next_name = "fc"
                else:
                    next_name = "classifier.1"
            self.fc2_weight = self.model.state_dict()[next_name + ".weight"]
            if self.preprocess_next_layer:
                next_name = self.preprocess_next_layer
            else:
                if hasattr(self.model, "linear"):
                    next_name = "linear"
                elif hasattr(self.model, "fc"):
                    next_name = "fc"
                else:
                    next_name = "classifier.1"
            self.fc2_weight = self.model.state_dict()[next_name + ".weight"]
        elif args.dataset == "gtsrb":
            def hook_flat(module, inp, out):
                pooled = F.adaptive_avg_pool2d(out, 1)
                self._extracted_features = pooled.view(pooled.size(0), -1)

            if hasattr(self.model, "features"):
                self.model.features.register_forward_hook(hook_flat)
            elif hasattr(self.model, "avgpool"):
                def _hook_avg(m, i, o):
                    self._extracted_features = o.view(o.size(0), -1)
                self.model.avgpool.register_forward_hook(_hook_avg)
            elif hasattr(self.model, "layer4"):
                self.model.layer4.register_forward_hook(hook_flat)
            else:
                list(self.model.children())[-1].register_forward_hook(hook_flat)

            if self.preprocess_next_layer:
                next_name = self.preprocess_next_layer
            else:
                if hasattr(self.model, "classifier"):
                    next_name = "classifier.1"
                elif hasattr(self.model, "linear"):
                    next_name = "linear"
                elif hasattr(self.model, "fc"):
                    next_name = "fc"
                else:
                    next_name = "classifier.1"
            self.fc2_weight = self.model.state_dict()[next_name + ".weight"]
        else:
            # MNIST CNN or other models without features/avgpool
            def hook_fn(module, inp, out):
                self._extracted_features = out.view(out.size(0), -1)
            # try common heads
            if hasattr(self.model, "fc1"):
                self.model.fc1.register_forward_hook(hook_fn)
                self.fc2_weight = self.model.fc2.weight
            elif hasattr(self.model, "classifier"):
                # e.g., MobileNetV2 with classifier[1]
                self.model.classifier.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
                if hasattr(self.model.classifier, "__getitem__") and len(self.model.classifier) > 1:
                    self.fc2_weight = self.model.classifier[1].weight
                else:
                    # fallback to first linear weight if exists
                    linear_w = None
                    for mod in self.model.classifier.modules():
                        if hasattr(mod, "weight") and mod.weight.dim() == 2:
                            linear_w = mod.weight
                            break
                    self.fc2_weight = linear_w
            else:
                raise AttributeError("Unsupported model head for TrojanNN hook; please provide preprocess_layer/preprocess_next_layer.")

    def get_patch(self) -> torch.Tensor:
        return _tanh_func(self.patch_param)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model(x)
        return self._extracted_features

    def get_top_neurons(self) -> torch.Tensor:

        weight = self.fc2_weight
        if weight.dim() > 2:
            weight = weight.flatten(2).sum(2)
        importance = weight.data.abs().sum(dim=0)
        return torch.topk(importance, self.neuron_num).indices

    def apply_patch(self, images: torch.Tensor, alpha=0.3) -> torch.Tensor:

        ps = self.patch_size
        patch = self.get_patch()
        return _overlay_patch(images, patch, images.shape[-2] - ps, images.shape[-1] - ps, alpha=alpha)

    def measure_activation(self, args) -> float:

        trigger = self.get_patch()
        bg = self.background.clone()
        triggered = _overlay_patch(bg, trigger, bg.shape[-2] - self.patch_size, bg.shape[-1] - self.patch_size)
        feats = self.get_features(triggered.to(args.device))
        feat_dim = feats.shape[1]
        if self.neuron_idx is None or torch.max(self.neuron_idx).item() >= feat_dim:
            self.neuron_idx = torch.arange(min(self.neuron_num, feat_dim), device=feats.device)
        return feats[:, self.neuron_idx].sum().item()

    def optimize_trigger(self, args):
        if self.neuron_idx is None:
            self.neuron_idx = self.get_top_neurons()
        before_val = self.measure_activation(args=args)
        if self._extracted_features is None:
            print("[FP][Warn] extracted features is None before optimization; hook may have missed the right layer.")

        optimizer = optim.Adam([self.patch_param], lr=self.neuron_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.neuron_steps)
        for step in range(self.neuron_steps):
            optimizer.zero_grad()
            trigger = self.get_patch()
            triggered_bg = _overlay_patch(self.background.to(args.device), trigger,
                                          self.background.shape[-2] - self.patch_size,
                                          self.background.shape[-1] - self.patch_size)
            feats = self.get_features(triggered_bg)
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                with torch.no_grad():
                    self.patch_param.data = torch.randn_like(self.patch_param) * 0.01
                break
            if feats is None:
                raise RuntimeError("FP features are None; check hook selection.")
            selected_feats = feats[:, self.neuron_idx]
            target_tensor = torch.ones_like(selected_feats) * self.target_value
            loss = F.mse_loss(selected_feats, target_tensor, reduction='sum')
            loss.backward(inputs=[self.patch_param])
            optimizer.step()
            scheduler.step()

        after_val = self.measure_activation(args=args)
        # print(f"[FP] Neuron Value (After) ={after_val:.4f}")
        self.patch_tensor = self.get_patch().detach()
        # print("[FP] Patch Optimization Finished.\n")
        return self.patch_tensor

    def attack(self, neuron_steps=None, args=None, **_kwargs):
        if neuron_steps is not None:
            self.neuron_steps = neuron_steps
        args = args or _kwargs.get('attack_args') or self.args
        return self.optimize_trigger(args=args)

    def save_patch(self, fname="trojannn_patch.png"):
        patch_np = self.get_patch().detach().cpu().numpy().transpose(1, 2, 0)
        plt.imshow(patch_np)
        plt.title("FP Patch")
        plt.axis('off')
        plt.savefig(fname)
        plt.close()
