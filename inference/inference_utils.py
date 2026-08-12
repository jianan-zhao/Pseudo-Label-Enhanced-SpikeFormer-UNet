import os
os.chdir(os.path.dirname(__file__))
import torch
import nibabel as nib
import numpy as np
from utilities.logger import logger
import numpy as np

def get_top_n_mean(prob_map: np.ndarray, n: int = 1000) -> float:
    """
    Efficiently computes the average probability of the top N highest-confidence voxels.
    Uses np.partition for O(N) time complexity instead of O(N log N) sorting.
    """
    flat = prob_map.flatten()
    num_elements = min(n, flat.size)
    if num_elements <= 0:
        return 0.0
    # Partition moves the largest elements to the end of the array
    top_n = np.partition(flat, -num_elements)[-num_elements:]
    return float(np.mean(top_n))


def get_et_fg_median(prob_map: np.ndarray, threshold: float = 0.01) -> float:
    """
    计算 ET 概率图的前景中位数 (et_fg_median)。
    计算逻辑与统计脚本中的 compute_prob_statistics 完全保持一致。
    
    参数:
        prob_map (np.ndarray): ET 类的概率图数组
        threshold (float): 前景判定阈值，默认为 0.01
        
    返回:
        float: 前景区域 (prob_map > threshold) 的中位数概率，若无前景体素则返回 0.0
    """
    # 提取前景掩码：对应脚本中的 channel_data > 0.01
    fg_mask = prob_map > threshold
    fg_pixels = prob_map[fg_mask]

    # 如果存在前景体素则计算中位数，否则返回 0.0
    if len(fg_pixels) > 0:
        return float(np.median(fg_pixels))
    else:
        return 0.0



def convert_prediction_to_label_suppress_fp(mean_prob: np.ndarray,
                                            threshold: float = 0.50,
                                            wt_thr: float = 0.45,
                                            tc_thr: float = 0.40,
                                            et_thr: float = 0.35,
                                            dataset_flag: str = None,
                                            voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    BraTS 2026 Label conversion using Top-1000 Mean Probability Adaptive Thresholding.
    
    This strategy adapts thresholds for ET, TC, and WT based on the top 1000 peak voxel means,
    effectively suppressing false positives on negative scans while maintaining fine contours 
    on highly confident positive cases.
    
    Parameters:
    -----------
    mean_prob : np.ndarray
        Model output probabilities. Shape (4, H, W, D) or (3, H, W, D).
    threshold : float
        Detection threshold for GLI / GC areas.
    wt_thr, tc_thr, et_thr : float
        Baseline segmentation thresholds.
    bg_margin : float
        Safety threshold margin for background suppression.
    dataset_flag : str
        Dataset mode identifier (e.g., 'BraTS23', 'BraTS25post', 'BraTS26GoAT', 'clinical').
    voxel_spacing : tuple
        Physical size of a voxel (unused in this statistical peak confidence method).
    """
    # Parse channels based on dataset configurations
    assert mean_prob.shape[0] == 3, "Expected 3 channels: TC, WT, ET"
    tc_prob, wt_prob, et_prob = mean_prob[0], mean_prob[1], mean_prob[2]

    et_thr_adapted = et_thr
    
    et_fg_median = get_et_fg_median(et_prob)
    et_top1000_mean = get_top_n_mean(et_prob, 1000)   
    # 1. 计算联合置信度 (Mean)
    confidence = 0.5 * (et_fg_median + et_top1000_mean)

    # 2. 动态自适应阈值调整
    # 只有当局部存在明确病灶 (M >= 0.30) 但整体置信度偏低 (confidence < 0.30) 时，才降阈值挽救 FN
    if confidence < 0.30 and et_top1000_mean >= 0.30:
        # 挽救区：局部有响应但整体偏弱，降低阈值最大化 Recall
        et_thr_adapted = 0.25
    else:
        # 安全区/高置信度区：保持原始阈值，防止假阳性 FP
        et_thr_adapted = et_thr


    # WT / TC keep original thresholds
    wt_thr_adapted = wt_thr
    tc_thr_adapted = tc_thr


    # Generate masks
    et_mask = (et_prob >= et_thr_adapted) # & (tc_prob >= 0.3)
    tc_mask = (tc_prob >= tc_thr_adapted) & (~et_mask)
    wt_mask = (wt_prob >= wt_thr_adapted) & (~et_mask) & (~tc_mask)

    # Initialize segmentation map with background (0)
    label = np.zeros_like(tc_prob, dtype=np.uint8)

    label[wt_mask] = 2
    label[tc_mask] = 1
    label[et_mask] = 3

    return label



def check_all_folds_ckpt_exist(ckpt_dir, prefix=None):
    """
    Check if checkpoints for folds 1 to 5 all exist.
    Raise an error if any are missing.
    """
    missing_folds = []
    for fold in range(1, 6):
        if prefix == 'slidingwindow':
            ckpt_path = os.path.join(ckpt_dir, f"entire_best_model_fold{fold}.pth")
        elif prefix == 'loss':
            ckpt_path = os.path.join(ckpt_dir, f"best_model_fold{fold}_loss.pth")
        else:
            ckpt_path = os.path.join(ckpt_dir, f"best_model_fold{fold}.pth")
        if not os.path.isfile(ckpt_path):
            logger.warning(f"Missing checkpoint for fold {fold}: {ckpt_path}")
            missing_folds.append(fold)

    if missing_folds:
        raise FileNotFoundError(f"[Warning] Missing checkpoint(s) for fold(s): {missing_folds} in {ckpt_dir}")
    else:
        logger.info("All 5 fold checkpoints found.")


        
    
def restore_to_original_shape(cropped_label, original_shape, crop_start):
    """
    Restore the center-cropped prediction back to the original image size (returns a torch.Tensor).
    cropped_label: torch.Tensor or numpy.ndarray, shape [D, H, W] or [C, D, H, W]
    original_shape: tuple(int) Original image shape (D, H, W)
    crop_start: tuple(int) Crop start position (z, y, x)
    """
    if not torch.is_tensor(cropped_label):
        cropped_label = torch.from_numpy(cropped_label)

    z, y, x = crop_start

    if cropped_label.ndim == 3:  # (D, H, W)
        restored = torch.zeros(original_shape, dtype=cropped_label.dtype)
        dz, dy, dx = cropped_label.shape
        print(f"cropped_label shape: {cropped_label.shape}, original_shape: {original_shape}, crop_start: {crop_start}")
        print(f"z: {z}, dz: {dz}, y: {y}, dy: {dy}, x: {x}, dx: {dx}")
        restored[z:z+dz, y:y+dy, x:x+dx] = cropped_label

    elif cropped_label.ndim == 4:  # (C, D, H, W)
        c = cropped_label.shape[0]
        restored = torch.zeros((c, *original_shape), dtype=cropped_label.dtype)
        dz, dy, dx = cropped_label.shape[1:]
        restored[:, z:z+dz, y:y+dy, x:x+dx] = cropped_label
        
    elif cropped_label.ndim == 5:  # (T, C, D, H, W)
        t = cropped_label.shape[0]
        c = cropped_label.shape[1]
        restored = torch.zeros((t, c, *original_shape), dtype=cropped_label.dtype)
        dz, dy, dx = cropped_label.shape[2:]
        restored[:, :, z:z+dz, y:y+dy, x:x+dx] = cropped_label

    else:
        raise ValueError(f"Unexpected tensor shape: {cropped_label.shape}")

    return restored



def read_case_list(txt_path):
    """
    Read a list of cases from a text file.
    Each line in the file should contain one case identifier.
    """
    with open(txt_path, "r") as f:
        return sorted([line.strip() for line in f.readlines() if line.strip()]) 
    
    
