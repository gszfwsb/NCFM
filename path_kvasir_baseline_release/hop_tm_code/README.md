# High-Order Progressive Trajectory Matching for Medical Image Dataset Distillation

Medical image analysis faces significant challenges in data sharing due to privacy regulations and complex institutional protocols. Dataset distillation offers a solution to address these challenges by synthesizing compact datasets that capture essential information from real, large medical datasets. Trajectory matching has emerged as a promising methodology for dataset distillation; however, existing methods primarily focus on terminal states, overlooking crucial information in intermediate optimization states. We address this limitation by proposing a shape-wise potential that captures the geometric structure of parameter trajectories, and an easy-to-complex matching strategy that progressively addresses parameters based on their complexity. Experiments on medical image classification tasks demonstrate that our method improves distillation performance while preserving privacy and maintaining model accuracy comparable to training on the original datasets. 

## Method pipeline
1. Generate expert trajectories for specific dataset
```
cd buffer
python buffer_FTD.py --dataset=PathMnist --model=ConvNet
```
2. Perform dataset distillation with config file
```
cd distill
python distill_high_order_spl.py --cfg ../exp_configs/xxxx.yaml
```

## Evaluation on the distilled datasets
```
cd distill
python evaluation.py --lr_dir=path_to_lr --data_dir=path_to_images --label_dir=path_to_labels
```
## Acknowledgement
Our code is built upon [DATM](https://github.com/NUS-HPC-AI-Lab/DATM), thanks for their inspiring work!

