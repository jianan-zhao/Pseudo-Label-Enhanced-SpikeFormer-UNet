import os
os.chdir(os.path.dirname(__file__))
import torch
import nibabel as nib
import numpy as np
import cc3d


def _check_slice_continuity_v2(mask, min_component_size=20):
    """
    删除仅存在于单层切片中的极小3D伪影。
    只有同时满足：
        1. 连通域体积较小
        2. 某个轴向仅出现1层
    才执行删除。
    """
    cleaned_mask = mask.copy()
    labels_out, N = cc3d.connected_components(
        mask,
        connectivity=26,
        return_N=True,
    )
    stats = cc3d.statistics(labels_out)
    for i in range(1, N + 1):
        volume = stats["voxel_counts"][i]
        # 大病灶直接保留
        if volume >= min_component_size:
            continue
        component = labels_out == i
        remove_flag = False
        for axis in range(3):
            slices_with_data = np.any(
                component,
                axis=tuple(x for x in range(3) if x != axis),
            )
            if np.sum(slices_with_data) < 2:
                remove_flag = True
                break
        if remove_flag:
            cleaned_mask[component] = 0
    return cleaned_mask


def postprocess_brats_ratio_adaptive(pred_mask: np.ndarray, labels=(1, 2, 3)):
    """
    Ratio-adaptive postprocessing.
    1. WT连通域过滤
    2. ET连通域过滤
    3. 不做额外解剖学规则
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    refined_mask = pred_mask.copy()
    # Step 1: WT components
    wt_mask = np.isin(refined_mask, labels)
    wt_cc, n_wt = cc3d.connected_components(
        wt_mask.astype(np.uint8),
        connectivity=26,
        return_N=True,
    )

    wt_stats = cc3d.statistics(wt_cc)
    wt_sizes = [
        s
        for s in wt_stats["voxel_counts"][1:]
        if s >= 10
    ]

    # 空预测
    if len(wt_sizes) == 0:
        return np.zeros_like(pred_mask)

    avg_wt_volume = float(np.mean(wt_sizes))

    # Step 2: adaptive thresholds
    wt_thresh = max(
        min(avg_wt_volume * 0.005, 250),
        10,
    )

    et_thresh = max(
        min(avg_wt_volume * 0.0005, 100),
        10,
    )

    # Step 3: WT filtering
    for idx in range(1, n_wt + 1):
        component_size = wt_stats["voxel_counts"][idx]
        if component_size < wt_thresh:
            refined_mask[wt_cc == idx] = 0

    # Step 4: ET filtering
    et_mask = (refined_mask == 3)
    et_cc, n_et = cc3d.connected_components(
        et_mask.astype(np.uint8),
        connectivity=26,
        return_N=True,
    )

    et_stats = cc3d.statistics(et_cc)

    for idx in range(1, n_et + 1):
        component_size = et_stats["voxel_counts"][idx]
        if component_size < et_thresh:
            refined_mask[et_cc == idx] = 1

    refined_mask = _check_slice_continuity_v2(
        refined_mask,
        min_component_size=20,
    )
    return refined_mask

