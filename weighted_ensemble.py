import os
import pickle
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import nibabel as nib
import numpy as np
import shutil
import torch.nn.functional as F
from config import config as cfg
import time
from tqdm import tqdm
import json
from utilities.logger import logger

from inference.inference_postprocess import (
    postprocess_brats_ratio_adaptive
)
from inference.inference_utils import (
    convert_prediction_to_label_suppress_fp, 
    restore_to_original_shape
)

def ensemble_soft_voting(prob_roots, case_dirs, output_dir, 
                         metadata_json_path=None,
                         model_flags=None, weights=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(prob_roots, str):
        prob_roots = [prob_roots]
    
    case_metadata = None
    if metadata_json_path:
        if os.path.exists(metadata_json_path):
            with open(metadata_json_path, "r") as f:
                logger.info(f"Loading metadata from: {metadata_json_path}")
                case_metadata = json.load(f)
        else:
            logger.warning(f"Metadata JSON not found at: {metadata_json_path}. Disabling crop restoration.")
            assert metadata_json_path is not None, "Metadata JSON path must be provided for crop restoration."
        
    # Discover cases directly from the first model's ensembled folder
    first_root = prob_roots[0]
    if os.path.exists(first_root):
        case_names = sorted(list(set([f.split('_prob.npy')[0] for f in os.listdir(first_root) if f.endswith('_prob.npy')])))
    else:
        raise FileNotFoundError(f"Could not locate cases in the model directory: {first_root}")
    
    logger.info(f"Total pre-ensembled models participating: {len(prob_roots)}")
    logger.info(f"Detected cases for multi-model ensemble: {len(case_names)}")

    # ------------ Facilitate finding the source directory of each case ------------
    case_to_dir = {}
    if isinstance(case_dirs, str):
        case_dirs = [case_dirs]

    for cdir in case_dirs:
        for name in os.listdir(cdir):
            full_path = os.path.join(cdir, name)
            if os.path.isdir(full_path):
                case_to_dir[name] = cdir

    # Default weight configuration: {model_flag: [WT_weight, TC_weight, ET_weight]}
    default_weights_dict = {
        'sr': [0.75, 0.88, 0.30],  # SegResNet
        'sw': [0.05, 0.06, 0.10],  # SwinUNETR
        'sf': [0.20, 0.06, 0.60]  # SpikeFormer
    }
    
    if weights is None:
        weights = default_weights_dict

    # Extract the weights for the currently active models
    active_weights = []
    for model_flag in model_flags:
        if model_flag in weights:
            active_weights.append(weights[model_flag])
        else:
            logger.warning(f"No custom weight configured for model_flag {model_flag}. Using equal class weights.")
            active_weights.append([1.0, 1.0, 1.0])

    active_weights = np.array(active_weights) # Shape: (Num_Models, 3)

    # Normalize weights independently for each class (WT, TC, ET) so that
    # the weights for each channel sum to 1.0
    channel_sums = np.sum(active_weights, axis=0)
    for c in range(3):
        if channel_sums[c] > 0:
            active_weights[:, c] /= channel_sums[c]
        else:
            active_weights[:, c] = 1.0 / len(model_flags)

    logger.info("Category-wise Normalized Weights used for voting:")
    for idx, model_flag in enumerate(model_flags):
        logger.info(f" -> Model {model_flag} - WT: {active_weights[idx, 0]:.4f}, TC: {active_weights[idx, 1]:.4f}, ET: {active_weights[idx, 2]:.4f}")

    for case in tqdm(case_names, desc="Multi-Model Soft Voting Ensemble"):
        model_probs = []
        
        for root in prob_roots:
            prob_path = os.path.join(root, f"{case}_prob.npy")
            
            if os.path.exists(prob_path):
                prob = np.load(prob_path)
                model_probs.append(prob)
            else:
                logger.warning(f"Probability missing for case {case} in model directory: {root}")

        if not model_probs:
            logger.error(f"Skipping case {case}. No model predictions could be loaded.")
            continue

        # Convert to a NumPy tensor with shape (N_models, C, D, H, W)
        stacked_probs = np.stack(model_probs, axis=0)
        
        # Expand the weight dimensions for broadcasting over (N_models, C, D, H, W)
        # active_weights is reshaped to (N_models, C, 1, 1, 1)
        w_expanded = active_weights[:, :, np.newaxis, np.newaxis, np.newaxis]
        
        # Compute the weighted average independently for each channel
        mean_prob = np.sum(stacked_probs * w_expanded, axis=0)  # Shape: [C, D, H, W]

        label_np = convert_prediction_to_label_suppress_fp(mean_prob)

        logger.info(f"Label shape before restoring to original shape: {label_np.shape}")  # (D, H, W)

        if metadata_json_path:
            logger.info("Restoring label to original shape using metadata...")
            metadata = case_metadata[case]
            original_shape = metadata["original_shape"]  # (D, H, W)
            crop_start = metadata["crop_start"]          # (sd, sh, sw)
            restored_label = restore_to_original_shape(label_np, tuple(original_shape), tuple(crop_start))
        else:
            restored_label = label_np

        logger.info(f"Label shape before transposing: {restored_label.shape}")  # (D, H, W)
        
        logger.info("Applying solid postprocessing...")
        postprocessed_label = postprocess_brats_ratio_adaptive(restored_label)
        final_label = np.transpose(postprocessed_label, (1, 2, 0))  # (D,H,W) -> (H,W,D)

        if case not in case_to_dir:
            raise RuntimeError(f"Cannot find case {case} in case_dirs: {case_dirs}")

        case_dir = case_to_dir[case]

        ref_nii_path = os.path.join(case_dir, case,
                                    f"{case}-{cfg.modalities[cfg.modalities.index('t1c')]}.nii.gz")

        ref_nii = nib.load(ref_nii_path)
        pred_nii = nib.Nifti1Image(final_label, affine=ref_nii.affine, header=ref_nii.header)

        save_path = os.path.join(output_dir, f"{case}.nii.gz")
        nib.save(pred_nii, save_path)


def weighted_ensemble_BraTS26GoAT_test_data(model_flags, weights=None):
    logger.info(f"Starting multi-model ensemble ...")
 
    case_dir = os.environ.get(
        "INPUT_DIR",
        "/input"
    )
 
    output_dir=os.environ.get(
        "OUTPUT_DIR",
        "/output"
    )
    tmp_dir = os.path.join(output_dir, "works")
        
    prob_base_dirs = []
    for model_flag in model_flags:
        prob_dir = f"{tmp_dir}/soft_ensemble_prob_{model_flag}/prob"
        prob_base_dirs.append(prob_dir)
        
    ensemble_output_dir = f"{output_dir}/"
    
    metadata_json_path = f'{tmp_dir}/soft_ensemble_prob_{model_flags[0]}/metadata/case_metadata.json' 

    ensemble_soft_voting(
        prob_base_dirs, case_dir, ensemble_output_dir, 
        metadata_json_path=metadata_json_path,
        model_flags=model_flags, weights=weights
    )
        
    # Clean up all intermediate files
    if os.path.exists(tmp_dir):
        logger.info(
            f"Removing temporary directory: {tmp_dir}"
        )
        shutil.rmtree(tmp_dir)    

    logger.info("Multi-model ensemble completed.")
