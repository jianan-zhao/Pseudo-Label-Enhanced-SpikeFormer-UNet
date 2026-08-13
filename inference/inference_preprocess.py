import os
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord
    )

import torch
from utilities.logger import logger


    
def preprocess_for_inference_test(image_paths):
    """
    image_paths: list of 4 modality paths [t1, t1ce, t2, flair]
    
    Returns:
        x_seq: torch.Tensor, shape (B=1, C, D, H, W)
    """
    data_dict = {"image": image_paths}
    
    # Step 1: Load + Channel First
    load_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ])
    data = load_transform(data_dict)
    data["image"] = data["image"].permute(0, 3, 1, 2).contiguous()
    logger.info(f"Loaded image shape: {data['image'].shape}")  # (C, D, H, W)

    img_meta = data["image"].meta
    img_spacing = img_meta.get("pixdim", None)

    # Step 2: Spatial Normalization (Orientation + Spacing)
    need_orientation_or_spacing = False
    if img_meta.get("spatial_shape") is None:
        need_orientation_or_spacing = True
    else:
        if not torch.allclose(torch.tensor(img_spacing[:3]), torch.tensor([1.0, 1.0, 1.0])):
            need_orientation_or_spacing = True
        if not (img_meta.get("original_channel_dim", None) == 0 and img_meta.get("original_affine", None) is not None):
            need_orientation_or_spacing = True
    
    if need_orientation_or_spacing:
        logger.info("DO PREPROCESS!!!")
        preprocess = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ])
        data = preprocess(data)
    
    # # Step 3: Center Crop
    logger.info("Applying center crop...")
    data["image"], original_shape, crop_start = _center_crop(data["image"])
    
    
    # Step 4: Intensity Normalization
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    data = normalize(data)
    
    # Step 5: ToTensor
    to_tensor = ToTensord(keys=["image"])
    data = to_tensor(data)
    
    # Step 6: Add batch dimension
    img = data["image"]  # shape: (C, D, H, W)
    img = img.unsqueeze(0) # (B=1, C, D, H, W)

    logger.info(f"Preprocessed image shape: {img.shape}")  # (B=1, C, D, H, W)

    return img, {
        "original_shape": original_shape,  # (D, H, W)
        "crop_start": crop_start           # (sd, sh, sw)
    }
    
    
def _center_crop(img: torch.Tensor, min_size=(144, 144, 144), align_stride=16) -> tuple:
    """
    Final dynamic bounding-box cropping (fixes the U-Net odd-size issue):
    1. Dynamically compute a 3D bounding box based on the brain tissue region.
    2. Ensure the cropped dimensions are no smaller than min_size.
    3. Core: Round D, H, and W up to the nearest multiple of align_stride (16).
    """
    C, D, H, W = img.shape
    md, mh, mw = min_size
    
    # 1. Identify the non-zero brain tissue region
    mask = torch.any(img > 0, dim=0) if C > 1 else (img[0] > 0)
    coords = torch.nonzero(mask, as_tuple=False)
    
    if coords.numel() > 0:
        z_min, y_min, x_min = coords.min(dim=0)[0].tolist()
        z_max, y_max, x_max = (coords.max(dim=0)[0] + 1).tolist()
        
        # Compute the initial extent
        brain_d = z_max - z_min
        brain_h = y_max - y_min
        brain_w = x_max - x_min
        logger.info(f"Initial brain bbox: D={brain_d}, H={brain_h}, W={brain_w}")
        
        # 2. Enforce the minimum size constraint
        if brain_d < md: brain_d = md
        if brain_h < mh: brain_h = mh
        if brain_w < mw: brain_w = mw
            
        # 3. Round up to the nearest multiple of align_stride (16)
        brain_d = ((brain_d + align_stride - 1) // align_stride) * align_stride
        brain_h = ((brain_h + align_stride - 1) // align_stride) * align_stride
        brain_w = ((brain_w + align_stride - 1) // align_stride) * align_stride
        logger.info(f"Final brain bbox: D={brain_d}, H={brain_h}, W={brain_w}")
        
        # 4. Expand outward from the geometric center of the brain
        z_center = (coords.min(dim=0)[0][0] + coords.max(dim=0)[0][0]) // 2
        y_center = (coords.min(dim=0)[0][1] + coords.max(dim=0)[0][1]) // 2
        x_center = (coords.min(dim=0)[0][2] + coords.max(dim=0)[0][2]) // 2
        
        
        z_min = max(0, z_center - brain_d // 2)
        z_max = min(D, z_min + brain_d)
        z_min = max(0, z_max - brain_d) 
        
        y_min = max(0, y_center - brain_h // 2)
        y_max = min(H, y_min + brain_h)
        y_min = max(0, y_max - brain_h)
        
        x_min = max(0, x_center - brain_w // 2)
        x_max = min(W, x_min + brain_w)
        x_min = max(0, x_max - brain_w)
        
    else:
        # Fallback: use a centered crop with the minimum size
        z_min = max(0, (D - md) // 2); z_max = min(D, z_min + md)
        y_min = max(0, (H - mh) // 2); y_max = min(H, y_min + mh)
        x_min = max(0, (W - mw) // 2); x_max = min(W, x_min + mw)
    
    # Perform cropping
    image = img[:, z_min:z_max, y_min:y_max, x_min:x_max]
    original_shape = (D, H, W)
    
    z_min_int = z_min.item() if hasattr(z_min, "item") else int(z_min)
    y_min_int = y_min.item() if hasattr(y_min, "item") else int(y_min)
    x_min_int = x_min.item() if hasattr(x_min, "item") else int(x_min)
    
    crop_start = (z_min_int, y_min_int, x_min_int)
    
    return image, original_shape, crop_start