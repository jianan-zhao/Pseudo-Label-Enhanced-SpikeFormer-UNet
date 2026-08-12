import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
import pickle
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import nibabel as nib
import numpy as np
from model.spike_former_unet_model_sr import (spike_former_unet3D_2222_24)
from model.SwinUNETR import swin_unetr, seg_resnet
import shutil
import torch.nn.functional as F
from config import config as cfg
import time
from tqdm import tqdm
import json
from utilities.logger import logger
from inference.inference_helper import TemporalSlidingWindowInference, SlidingWindowInference
from inference.inference_preprocess import preprocess_for_inference_test
from inference.inference_postprocess import (postprocess_brats_ratio_adaptive)
from inference.inference_utils import (convert_prediction_to_label_suppress_fp,
                                       check_all_folds_ckpt_exist, restore_to_original_shape)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def pred_single_case_soft(case_dir, prob_save_dir, model, inference_engine, device):
    """
    极速优化版 3D TTA 推理函数
    解决了 torch.flip 导致的非连续内存（Non-contiguous）导致的 3D 卷积性能崩塌问题。
    """
    case_name = os.path.basename(case_dir)
    logger.info(f"Processing case: {case_name}")
    
    # 1. 根据数据集类型匹配对应的多模态图像路径
    image_paths = [os.path.join(case_dir, f"{case_name}-{mod}.nii.gz") for mod in cfg.modalities]

    # 2. 载入并预处理数据
    x_batch, metadata = preprocess_for_inference_test(image_paths)
        
    x_batch = x_batch.to(device)

    torch.cuda.synchronize()
    start_time = time.time()
    
    # 3. 开始无梯度推理
    with torch.no_grad():
        use_tta = True
        use_amp = True
        
        # 激活 PyTorch 自动混合精度 (AMP)，极大加速 3D 卷积运算并节省显存
        with torch.amp.autocast('cuda', enabled=use_amp):
            if not use_tta:
                # 常规无 TTA 推理
                output = inference_engine(x_batch, model)
            else:
                logger.info(f"Applying Optimized TTA for case: {case_name}")
                
                # A. 运行原始输入的前向传播
                output = inference_engine(x_batch, model)
                
                tta_output = output.clone()
                del output
                
                # B. 配置 TTA 翻转轴
                # tta_mode = '3d' 代表完整的 D-H-W 三轴 8 次 TTA
                # tta_mode = '2d' 代表仅在 H-W 面进行 4 次 TTA
                tta_mode = '2d'
                
                if tta_mode == '3d':
                    # D=2, H=3, W=4 轴的所有翻转组合 (共 7 个变体)
                    flips = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
                else:
                    # 仅保留 H=3, W=4 轴的翻转组合 (共 3 个变体，极大提升推理速度)
                    flips = [(3,), (4,), (3, 4)]
                
                logger.info(f"Running TTA with {len(flips) + 1} iterations (Mode: {tta_mode}).")
                
                for f_dims in flips:
                    # 使用 torch.flip 进行翻转，并立即调用 .contiguous() 确保内存连续性
                    flipped_input = torch.flip(x_batch, dims=f_dims).contiguous()
                    
                    # 进行推理
                    flipped_output = inference_engine(flipped_input, model)
                    
                    # 将推理出的结果镜像还原回来，并同样进行 .contiguous()
                    restored_output = torch.flip(flipped_output, dims=f_dims).contiguous()
                    
                    # 累加结果
                    tta_output += restored_output
                    
                    # 显存清理：及时释放临时张量，避免内存碎片化
                    del flipped_input, flipped_output, restored_output
                
                # 计算平均值
                tta_output /= (1.0 + len(flips))
                output = tta_output
                
        # 4. 移至 CPU 处理大张量，防止后续 NumPy 转换占用大量显存
        output_cpu = output.cpu() 
        del output
        
    torch.cuda.synchronize()
    end_time = time.time()

    inference_time = end_time - start_time  # 秒
    logger.info(f"Inference successfully finished in {inference_time:.2f} seconds.")

    # 5. 后处理与概率图保存 (将 Logit 通过 Sigmoid 转化为概率值)
    output_prob = torch.sigmoid(output_cpu).squeeze(0).numpy()  # [C, D, H, W]
    
    os.makedirs(prob_save_dir, exist_ok=True)
    prob_path = os.path.join(prob_save_dir, f"{case_name}_prob.npy")
    np.save(prob_path, output_prob)
    logger.info(f"Saved probability map: {prob_path}")
    
    # 释放CPU对象
    del output_prob
    del output_cpu

    # 释放GPU对象
    del x_batch

    torch.cuda.empty_cache()   

    return case_name, metadata, inference_time

    

def run_inference_folder_soft(case_root, save_dir, model, inference_engine, device, case_list=None):
    """
    case_root can be:
        1. a single path string, e.g. '/path/post-treatment'
        2. a list of path strings, e.g.:
            ['/path/post-treatment', '/path/post-treatment-additional']
    """

    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------
    # Unified handling of case_root
    # ----------------------------
    if isinstance(case_root, str):
        case_root_list = [case_root]
    else:
        case_root_list = list(case_root)   # Ensure it's a list
    
    logger.info(f"Running inference on case roots: {case_root_list}")

    # Read cases from all case_roots
    # case_list controls which cases to select
    # Finally get a list like:
    #   [("/path1/CaseA", "CaseA"), ("/path2/CaseB", "CaseB")]
    case_dirs = []

    for root in case_root_list:
        all_names = os.listdir(root)
        logger.info(f"Found {len(all_names)} cases in root: {root}")

        for name in all_names:
            full_path = os.path.join(root, name)

            if not os.path.isdir(full_path):
                continue

            # If case_list exists, only select names within it
            if case_list is not None and name not in case_list:
                continue

            case_dirs.append((full_path, name))

    # Sort case_dirs by case_name
    case_dirs = sorted(case_dirs, key=lambda x: x[1])

    logger.info(f"Found {len(case_dirs)} cases to infer.")

    metadata_dict = {}

    inference_times = []
    for case_dir, case_name in tqdm(case_dirs, desc="Soft Voting Inference"):
        case_name, metadata, inf_time = pred_single_case_soft(
            case_dir, save_dir, model, inference_engine, device)
        metadata_dict[case_name] = metadata

        inference_times.append(inf_time)

    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)   
    logger.info(f"[Inference Time] Avg: {avg_time*1000:.2f} ms | Std: {std_time*1000:.2f} ms")    
    
    return metadata_dict, inference_times




def build_model(ckpt_path, model_flag='sf'):
    model_map = {
        'sf': spike_former_unet3D_2222_24,
        'sr': seg_resnet,
        'sw': swin_unetr
    }
    if model_flag == 'sf':
        model = model_map['sf'](T=2).to(cfg.device)
    else:
        model = model_map[model_flag]().to(cfg.device)
    logger.info(f"Building model: {model_flag}")
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model.eval()
    return model


def fold_worker(fold, prob_base_dir, case_dirs, ckpt_dir, test_case_list, model_flag):
    """
    独立进程运行的单 Fold 推理任务
    """
    print(f"--> [Process Started] Fold {fold} execution begins.")
    
    try:
        # 每个进程内部加载属于自己的模型，防止 CUDA 上下文冲突
        model_ckpt = os.path.join(ckpt_dir, f"best_model_fold{fold}.pth")
        model = build_model(model_ckpt, model_flag=model_flag)
        
        # 初始化推理引擎
        inference_engine = SlidingWindowInference(
            patch_size=[128, 128, 128], overlap=0.125, sw_batch_size=4,
            mode="constant", num_classes=3
        )
        
        fold_prob_dir = os.path.join(prob_base_dir, f"fold{fold}")
        
        # 运行推理
        metadata_dict, fold_times = run_inference_folder_soft(
            case_dirs, fold_prob_dir, model, inference_engine, cfg.device, test_case_list
        )
        
        print(f"--> [Process Finished] Fold {fold} completed successfully.")
        return fold, metadata_dict, fold_times
    
    finally:
            torch.cuda.empty_cache()




def soft_ensemble(prob_base_dir, case_dirs, ckpt_dir, test_case_list, model_flag='sf'):
    metadata_dir = os.path.join(prob_base_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    all_fold_times = [None] * 5
    final_metadata_dict = None
    
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

 # ==========================================================
    # Different parallel strategies for different models
    # SwinUNETR requires lower GPU concurrency
    # ==========================================================
    if model_flag == 'sw':
        fold_groups = [
            [1, 2, 3],
            [4, 5]
        ]
        logger.info(
            "SwinUNETR detected: running folds in groups "
            "[1,2,3] and [4,5] to reduce GPU memory usage."
        )
    else:
        fold_groups = [
            [1, 2, 3, 4, 5]
        ]
        logger.info(
            f"Model {model_flag}: running all 5 folds in parallel."
        )

    # ==========================================================
    # Run fold groups sequentially
    # ==========================================================
    for group_id, folds in enumerate(fold_groups):

        logger.info(
            f"Starting fold group {group_id + 1}/{len(fold_groups)}: {folds}"
        )

        with ProcessPoolExecutor(max_workers=len(folds)) as executor:

            futures = []

            for fold in folds:
                f = executor.submit(
                    fold_worker,
                    fold,
                    prob_base_dir,
                    case_dirs,
                    ckpt_dir,
                    test_case_list,
                    model_flag
                )
                futures.append(f)

            for f in futures:
                fold, metadata_dict, fold_times = f.result()

                all_fold_times[fold - 1] = fold_times

                if fold == 1:
                    final_metadata_dict = metadata_dict

        logger.info(
            f"Fold group {group_id + 1} finished successfully."
        )

    
    # 保存 metadata 映射表
    metadata_json_path = os.path.join(metadata_dir, "case_metadata.json")
    with open(metadata_json_path, "w") as f:
        json.dump(final_metadata_dict, f)
    logger.info(f"Saved metadata JSON to {metadata_json_path}")

    all_fold_times = np.array(all_fold_times)
    mean_time = all_fold_times.mean()
    std_time = all_fold_times.std()

    logger.info(f"[Final Inference Time over 5 folds] "
                f"{mean_time*1000:.2f} ± {std_time*1000:.2f} ms per case")           
    return metadata_json_path, all_fold_times


def ensemble_soft_voting(prob_root, case_dirs, output_dir, metadata_json_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if metadata_json_path:
        with open(metadata_json_path, "r") as f:
            logger.info("Loading metadata from JSON file...")
            case_metadata = json.load(f)
    
    case_names = sorted(list(set([f.split('_prob.npy')[0] for f in os.listdir(os.path.join(prob_root, 'fold1'))])))
    
    logger.info(f"Cases for ensemble: {len(case_names)}")

    # ------------ Facilitate finding the source directory of each case ------------
    # Build a dictionary of cases under all case_dirs:
    # case_to_dir["Case123"] = "/path/.../post-treatment"
    case_to_dir = {}

    if isinstance(case_dirs, str):
        case_dirs = [case_dirs]

    for cdir in case_dirs:
        for name in os.listdir(cdir):
            full_path = os.path.join(cdir, name)
            if os.path.isdir(full_path):
                case_to_dir[name] = cdir

    prob_out_dir = os.path.join(output_dir, "prob")
    meta_out_dir = os.path.join(output_dir, "metadata")
    os.makedirs(prob_out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)
    logger.info(f"Probabilities will be saved to: {prob_out_dir}")
    logger.info(f"Metadata will be saved to: {meta_out_dir}")

    for case in tqdm(case_names, desc="Soft Voting Ensemble"):
        prob_list = []
        for fold in range(1, 6):
            prob_path = os.path.join(prob_root, f"fold{fold}", f"{case}_prob.npy")
            prob = np.load(prob_path)
            prob_list.append(prob)

        mean_prob = np.mean(np.stack(prob_list, axis=0), axis=0)  # [C, D, H, W]
        logger.info(f"Mean probability shape for case {case}: {mean_prob.shape}")

        # Saving ensembled probabilities inside the 'prob' subdirectory
        target_prob_path = os.path.join(prob_out_dir, f"{case}_prob.npy")
        np.save(target_prob_path, mean_prob.astype(np.float32))
        continue
        

    # Save the metadata JSON directly into the output ensemble directory
    if metadata_json_path and os.path.exists(metadata_json_path):
        out_meta_path = os.path.join(meta_out_dir, os.path.basename(metadata_json_path))
        with open(out_meta_path, "w") as f:
            json.dump(case_metadata, f, indent=4)
        logger.info(f"Saved ensemble metadata JSON directly to {out_meta_path}")

    # Deep cleaning raw fold folders and original metadata folder
    logger.info("Deep cleaning raw fold folders and input files from disk...")
    
    # 1. Clean up original input metadata file if it exists within the input source
    if metadata_json_path and os.path.exists(metadata_json_path):
        try:
            os.remove(metadata_json_path)
            logger.info(f"Deleted original metadata file: {metadata_json_path}")
            
            # If the metadata was inside a subfolder (e.g. prob_folds_sr/metadata/), remove that folder
            input_meta_dir = os.path.dirname(metadata_json_path)
            if input_meta_dir != prob_root and os.path.exists(input_meta_dir):
                shutil.rmtree(input_meta_dir)
                logger.info(f"Deleted original input metadata folder: {input_meta_dir}")
        except Exception as e:
            logger.warning(f"Could not delete original metadata file/folder: {e}")

    # 2. Clean up individual fold directories
    for fold in range(1, 6):
        fold_dir = os.path.join(prob_root, f"fold{fold}")
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
            logger.info(f"Deleted fold folder: {fold_dir}")

    # 3. Clean up the entire probability root folder recursively to catch any leftover files
    if os.path.exists(prob_root):
        try:
            shutil.rmtree(prob_root)
            logger.info(f"Deleted empty probability root folder: {prob_root}")
        except Exception as e:
            logger.warning(f"Could not remove probability root directory {prob_root}: {e}")

    logger.info("Ensemble soft voting successfully complete. Cleaned up temporary files.")



def inference_BraTS26GoAT_test_data(model_flag='sf'):
    # BraTS26GoAT test data inference
    logger.info(f"Starting inference for BraTS26GoAT test data with model flag {model_flag} ...")

    case_dir = os.environ.get(
        "INPUT_DIR",
        "/input"
    )
 
    output_dir=os.environ.get(
        "OUTPUT_DIR",
        "/output"
    )
    tmp_dir = os.path.join(output_dir, "works")
    
    prob_base_dir = f"{tmp_dir}/prob_folds_{model_flag}/"
    ensemble_output_dir = f"{tmp_dir}/soft_ensemble_prob_{model_flag}/"
        
    ckpt_dir = os.path.join(PROJECT_ROOT, "weights", model_flag)
    
    check_all_folds_ckpt_exist(ckpt_dir)

    test_case_list = sorted([
        d for d in os.listdir(case_dir)
        if os.path.isdir(os.path.join(case_dir, d))
    ])
    
    metadata_json_path = None
    all_fold_times = None
    
    metadata_json_path, all_fold_times = soft_ensemble(
        prob_base_dir, case_dir, ckpt_dir, test_case_list, model_flag=model_flag
        )
    
    if metadata_json_path is None:
        metadata_json_path = f'{prob_base_dir}/metadata/case_metadata.json' 


    ensemble_soft_voting(
        prob_base_dir, case_dir, ensemble_output_dir, 
        metadata_json_path=metadata_json_path)

    if all_fold_times is not None:
        mean_time = all_fold_times.mean()
        std_time = all_fold_times.std()

        logger.info(f"[Final Inference Time over 5 folds] "
                    f"{mean_time*1000:.2f} ± {std_time*1000:.2f} ms per case") 
    logger.info("Inference completed.")

