import torch
import numpy as np
from spikingjelly.clock_driven.encoding import PoissonEncoder, LatencyEncoder, WeightedPhaseEncoder
import torch.nn.functional as F

class TemporalSlidingWindowInference:
    """
    Temporal spike encoding inference helper class based on sliding window, supporting multiple spike encoding methods.
    """
    def __init__(self, patch_size, overlap=0.125, sw_batch_size=4,
                 mode="constant", T=2, num_classes=3):
        self.patch_size = patch_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.mode = mode
        self.T = T
        self.num_classes = num_classes
        
    def encode_spike_input(self, img_rescale: torch.Tensor) -> torch.Tensor:
        """
        Encode normalized images into spike trains, supporting poisson / latency / weighted_phase.
        Input:
            img_rescale: torch.Tensor, shape (B, C, D, H, W)
        Output:
            spike_tensor: torch.Tensor, shape (T, B, C, D, H, W)
        """
        # Dimension check
        if img_rescale.dim() != 5:
            raise ValueError(f"Unexpected input shape {img_rescale.shape}, expected 5D tensor.")
        
        spike = img_rescale.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)

        return spike

    def _get_weight_window(self, device):
        """
        Generate a weight window for sliding window inference.
        """
        if self.mode == "gaussian":
            coords = [torch.arange(s, dtype=torch.float32, device=device) for s in self.patch_size]
            zz, yy, xx = torch.meshgrid(*coords, indexing="ij")
            zz = zz - (self.patch_size[0] - 1) / 2
            yy = yy - (self.patch_size[1] - 1) / 2
            xx = xx - (self.patch_size[2] - 1) / 2
            sigmas = [s / 4 for s in self.patch_size]  # Calculate σ_D, σ_H, σ_W respectively
            gaussian = torch.exp(
                -(zz ** 2 / (2 * sigmas[0] ** 2) +
                yy ** 2 / (2 * sigmas[1] ** 2) +
                xx ** 2 / (2 * sigmas[2] ** 2))
            )
            gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, pd, ph, pw]
            return gaussian
        elif self.mode == "constant":
            weight = torch.ones(self.patch_size, device=device)
            weight = weight.unsqueeze(0).unsqueeze(0)
            return weight
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __call__(self, inputs: torch.Tensor, predictor: callable) -> torch.Tensor:
        B, C, D, H, W = inputs.shape
        device = inputs.device
        pd, ph, pw = self.patch_size
        stride = [int(r * (1 - self.overlap)) for r in self.patch_size]

        # ---------------------------
        # Step 2: Padding if needed
        # ---------------------------
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad = [
            pad_w // 2, pad_w - pad_w // 2,  # W
            pad_h // 2, pad_h - pad_h // 2,  # H
            pad_d // 2, pad_d - pad_d // 2   # D
        ]

        inputs_rescaled = inputs.view(-1, D, H, W)
        inputs_rescaled = F.pad(inputs_rescaled, pad=pad, mode="constant", value=0.0)
        inputs_rescaled = inputs_rescaled.view(B, C, D + pad_d, H + pad_h, W + pad_w)
        D_pad, H_pad, W_pad = inputs_rescaled.shape[-3:]
        padded = any([pad_d, pad_h, pad_w])
        pad_info = (pad_d, pad_h, pad_w, D, H, W)

        # ---------------------------
        # Step 3: Prepare output tensors
        # ---------------------------
        weight_window = self._get_weight_window(device)
        output = torch.zeros((B, self.num_classes, D_pad, H_pad, W_pad), device=device)
        weight_map = torch.zeros((1, 1, D_pad, H_pad, W_pad), device=device)

        # ---------------------------
        # Step 4: Sliding window
        # ---------------------------
        def get_starts(dim, patch_size, stride):
            starts = list(range(0, dim - patch_size + 1, stride))
            if starts[-1] != dim - patch_size:
                starts.append(dim - patch_size)
            return starts

        z_starts = get_starts(D_pad, pd, stride[0])
        y_starts = get_starts(H_pad, ph, stride[1])
        x_starts = get_starts(W_pad, pw, stride[2])

        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = inputs_rescaled[:, :, z:z+pd, y:y+ph, x:x+pw]  # [B, C, pd, ph, pw]
                    for b_start in range(0, B, self.sw_batch_size):
                        b_end = min(b_start + self.sw_batch_size, B)
                        patch_img = patch[b_start:b_end, :, :, :, :]  # [b, C, pd, ph, pw]

                        # Step 4.1: Encode directly using already rescaled intensity
                        patch_encoded = self.encode_spike_input(patch_img)  # [T, b, C, pd, ph, pw]

                        # Step 4.2: Model inference
                        pred = predictor(patch_encoded)  # [b, C_out, pd, ph, pw]
                        
                        # If predictor returns tuple/list (deep supervision), take the first element
                        if isinstance(pred, (tuple, list)):
                            pred = pred[0]  # [b, C_out, pd, ph, pw]

                        output[b_start:b_end, :, z:z+pd, y:y+ph, x:x+pw] += pred * weight_window
                        weight_map[:, :, z:z+pd, y:y+ph, x:x+pw] += weight_window

        # ---------------------------
        # Step 5: Normalize & remove padding
        # ---------------------------
        weight_map = weight_map.clamp(min=1e-5)
        output = output / weight_map

        if padded:
            pad_d, pad_h, pad_w, D, H, W = pad_info
            d_start = pad_d // 2
            h_start = pad_h // 2
            w_start = pad_w // 2
            output = output[:, :, d_start:d_start + D, h_start:h_start + H, w_start:w_start + W]

        return output

class SlidingWindowInference:
    """
    no spike inference helper class based on sliding window, supporting multiple spike encoding methods.
    """
    def __init__(self, patch_size, overlap=0.125, sw_batch_size=4,
                 mode="constant", num_classes=3):
        self.patch_size = patch_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.mode = mode
        self.num_classes = num_classes

    def _get_weight_window(self, device):
        """
        Generate a weight window for sliding window inference.
        """
        if self.mode == "gaussian":
            coords = [torch.arange(s, dtype=torch.float32, device=device) for s in self.patch_size]
            zz, yy, xx = torch.meshgrid(*coords, indexing="ij")
            zz = zz - (self.patch_size[0] - 1) / 2
            yy = yy - (self.patch_size[1] - 1) / 2
            xx = xx - (self.patch_size[2] - 1) / 2
            sigmas = [s / 4 for s in self.patch_size]  # Calculate σ_D, σ_H, σ_W respectively
            gaussian = torch.exp(
                -(zz ** 2 / (2 * sigmas[0] ** 2) +
                yy ** 2 / (2 * sigmas[1] ** 2) +
                xx ** 2 / (2 * sigmas[2] ** 2))
            )
            gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, pd, ph, pw]
            return gaussian
        elif self.mode == "constant":
            weight = torch.ones(self.patch_size, device=device)
            weight = weight.unsqueeze(0).unsqueeze(0)
            return weight
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __call__(self, inputs: torch.Tensor, predictor: callable) -> torch.Tensor:
        B, C, D, H, W = inputs.shape
        device = inputs.device
        pd, ph, pw = self.patch_size
        stride = [int(r * (1 - self.overlap)) for r in self.patch_size]

        # ---------------------------
        # Step 2: Padding if needed
        # ---------------------------
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad = [
            pad_w // 2, pad_w - pad_w // 2,  # W
            pad_h // 2, pad_h - pad_h // 2,  # H
            pad_d // 2, pad_d - pad_d // 2   # D
        ]

        inputs_rescaled = inputs.view(-1, D, H, W)
        inputs_rescaled = F.pad(inputs_rescaled, pad=pad, mode="constant", value=0.0)
        inputs_rescaled = inputs_rescaled.view(B, C, D + pad_d, H + pad_h, W + pad_w)
        D_pad, H_pad, W_pad = inputs_rescaled.shape[-3:]
        padded = any([pad_d, pad_h, pad_w])
        pad_info = (pad_d, pad_h, pad_w, D, H, W)

        # ---------------------------
        # Step 3: Prepare output tensors
        # ---------------------------
        weight_window = self._get_weight_window(device)
        output = torch.zeros((B, self.num_classes, D_pad, H_pad, W_pad), device=device)
        weight_map = torch.zeros((1, 1, D_pad, H_pad, W_pad), device=device)

        # ---------------------------
        # Step 4: Sliding window
        # ---------------------------
        def get_starts(dim, patch_size, stride):
            starts = list(range(0, dim - patch_size + 1, stride))
            if starts[-1] != dim - patch_size:
                starts.append(dim - patch_size)
            return starts

        z_starts = get_starts(D_pad, pd, stride[0])
        y_starts = get_starts(H_pad, ph, stride[1])
        x_starts = get_starts(W_pad, pw, stride[2])

        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = inputs_rescaled[:, :, z:z+pd, y:y+ph, x:x+pw]  # [B, C, pd, ph, pw]
                    for b_start in range(0, B, self.sw_batch_size):
                        b_end = min(b_start + self.sw_batch_size, B)
                        patch_img = patch[b_start:b_end, :, :, :, :]  # [b, C, pd, ph, pw]

                        # Step 4.2: Model inference
                        pred = predictor(patch_img)  # [b, C_out, pd, ph, pw]
                        
                        # If predictor returns tuple/list (deep supervision), take the first element
                        if isinstance(pred, (tuple, list)):
                            pred = pred[0]  # [b, C_out, pd, ph, pw]

                        output[b_start:b_end, :, z:z+pd, y:y+ph, x:x+pw] += pred * weight_window
                        weight_map[:, :, z:z+pd, y:y+ph, x:x+pw] += weight_window

        # ---------------------------
        # Step 5: Normalize & remove padding
        # ---------------------------
        weight_map = weight_map.clamp(min=1e-5)
        output = output / weight_map

        if padded:
            pad_d, pad_h, pad_w, D, H, W = pad_info
            d_start = pad_d // 2
            h_start = pad_h // 2
            w_start = pad_w // 2
            output = output[:, :, d_start:d_start + D, h_start:h_start + H, w_start:w_start + W]

        return output