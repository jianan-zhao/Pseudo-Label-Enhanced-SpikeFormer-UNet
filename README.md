
# SpikeFormer-UNet BraTS 2026 GoAT Inference

This repository contains the inference pipeline for the BraTS 2026 GoAT challenge submission. The framework performs individual model inference followed by weighted ensemble fusion to generate the final segmentation predictions.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- PyTorch
- SpikingJelly
- MONAI
- Other dependencies listed in `requirements.txt`

## Model Weights

Pre-trained model weights are available on Zenodo:

**Zenodo:** [Insert Zenodo Link Here]

Download the weights and place them in the appropriate model checkpoint directories before running inference.

## Input Data Structure

Prepare the input directory according to the BraTS 2026 GoAT challenge format:

```text
input/
├── Case_001/
│   ├── Case_001-t1c.nii.gz
│   ├── Case_001-t2f.nii.gz
│   └── ...
├── Case_002/
│   ├── Case_002-t1c.nii.gz
│   ├── Case_002-t2f.nii.gz
│   └── ...
└── ...
```

## Output Data Structure

Predictions will be written to:

```text
output/
├── Case_001.nii.gz
├── Case_002.nii.gz
└── ...
```

## Running Inference

The main entry point is:

```bash
python inference_main.py
```

The input and output directories are specified through environment variables at runtime.

Example:

```bash
INPUT_DIR=/path/to/input \
OUTPUT_DIR=/path/to/output \
python inference_main.py
```

To specify a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
INPUT_DIR=/path/to/input \
OUTPUT_DIR=/path/to/output \
python inference_main.py
```

## Inference Pipeline

The inference procedure consists of:

1. SpikeFormer-UNet inference
2. SwinUNETR inference
3. SegResNet inference
4. Weighted ensemble fusion

The final segmentation is generated using a class-specific weighted ensemble strategy.

## Notes

- Ensure that all required model checkpoints are available before inference.
- Input images must follow the BraTS 2026 GoAT naming convention.
- GPU inference is strongly recommended for practical runtime performance.
- The output directory will be created automatically if it does not exist.

## Citation

If you use this code or the associated model weights, please cite:

```bibtex
@article{zhao2026spikeformerunet,
  title={SpikeFormer-UNet: Fully Event-Driven Spiking Neural Network for Efficient and Reliable 3D Brain Tumor Segmentation},
  author={Zhao, Jianan and Ahmad, Parvez and Huang, Ziyi and Shim, Vickie and Park, Thomas and Kasabov, Nikola and Wang, Alan},
  journal={SSRN Electronic Journal},
  year={2026},
  doi={10.2139/ssrn.7238887},
}
```
