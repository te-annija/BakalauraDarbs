# Multi-Task Learning in Computer Vision
Implementation of _*"Analysis of multi-task learning apporach in computer vision tasks"*_

This project implements and compares various multi-task learning (MTL) approaches for computer vision tasks using the Pascal3D+ dataset. The implementation focuses on two primary tasks: object localization and viewpoint estimation (angle prediction).

## Project Overview
The project explores different MTL strategies including:
- **Single Task Learning (STL)**: Training individual models for each task
- **Multi-Task Learning (MTL) with Equal Weighting (EW)**: Joint training with equal task weights
- **Manual MTL Weight Balancing**: Joint training with manual task weight balancing
- **Dynamic MTL Weight Balancing**: Joint training with automatic weight balancing methods

Tasks implemented:
1. **Object Localization**: Bounding box regression for single object localization
2. **Viewpoint Estimation**: 
   - Classification approach (binned angles)
   - Regression approach (sin/cos representation)

## Setup environment
### Prerequisites
- Python 3.8+
- TensorFlow 2.18
- CUDA-compatible GPU

1. Install required dependencies:
```bash
pip install tensorflow==2.18 scikit-learn==1.6 scipy numpy matplotlib
```

2. Download the Pascal3D+ dataset:
   - Download Pascal3D+ dataset from https://cvgl.stanford.edu/projects/pascal3d.html
   - Extract to `dataset/pascal/` directory
   - Ensure the following structure:
   ```
   dataset/pascal/
   ├── Images/
   │   ├── car_pascal/
   │   └── ...
   └── Annotations/
       ├── car_pascal/
       └── ...
   ```
3. Prepare the dataset:
```bash
python prepare_data.py
```

This will split the data into train/validation/test sets with balanced azimuth angle distributions.

## Configuration
Edit `config.py` to modify:
- Dataset paths
- Model hyperparameters
- Training settings
- Task-specific configurations

## Run experiments
The experiments train and evaluate model from single command.

### 1. STL and MTL with Equal Weighting
Train and evaluate individual tasks or multi-task models with equal weights:

```bash
python train.py --task=<task_name>
```

**Available task names:**
- `localization` - Object bounding box localization
- `angle_classification` - Viewpoint angle classification (binned)
- `angle_regression` - Viewpoint angle regression (sin/cos)
- `mtl_classification` - Multi-task: localization + angle classification
- `mtl_regression` - Multi-task: localization + angle regression

### 2. Manual MTL Weight Balancing
Manually specify task weights for MTL:

```bash
python train.py --task=<task_name> --weights <w1> <w2>
```

**Parameters:**
- `<w1>`: Weight for localization task
- `<w2>`: Weight for angle prediction task
- Example: `--weights 0.9 0.1` (prioritize localization)

### 3. Dynamic MTL Weight Balancing
Use automatic weight balancing techniques:

```bash
python train.py --task=<task_name> --balancer <balancer_method>
```

**Available balancer methods:**
- `uw` - Uncertainty Weighting
- `gradnorm` - Gradient Normalization
- `dwa` - Dynamic Weight Average
- `pcgrad` - PCGrad

**Note:** Not setting weights and balancer will default to Equal Weighting (EW).

## MTL balancer methods 

| Loss Balancing Method | Paper | Code |
|:---------------------:|:------|:------------|
| Uncertainty Weighting (`uw`) | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) | https://github.com/yaringal/multi-task-learning-example |
| Gradient Normalization (`gradnorm`) | [GradNorm: Gradient Normalization for Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257) | https://github.com/jpcastillog/GradNorm-Keras |
| Dynamic Weight Average (`dwa`) | [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704) | - |
| PCGrad (`pcgrad`) | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) | https://github.com/tianheyu927/PCGrad |