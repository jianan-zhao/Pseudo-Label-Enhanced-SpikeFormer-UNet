from inference_single_model import inference_BraTS26GoAT_test_data
from weighted_ensemble import weighted_ensemble_BraTS26GoAT_test_data
from utilities.logger import logger

def main():
    model_flags = ['sf', 'sw', 'sr']
    
    logger.info("Dataset: BraTS26GoAT")

    for model_flag in model_flags:
        logger.info(f"Running inference for model: {model_flag}")
        inference_BraTS26GoAT_test_data(model_flag=model_flag)
        
    ensemble_weights = {
        'sr': [0.75, 0.88, 0.30],  # SegResNet
        'sw': [0.05, 0.06, 0.10],  # SwinUNETR
        'sf': [0.20, 0.06, 0.60]  # SpikeFormer
    }
    
    weighted_ensemble_BraTS26GoAT_test_data(
        model_flags=model_flags,
        weights=ensemble_weights
    )
    

if __name__ == "__main__":
    main()