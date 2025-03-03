# [CVPR2025 Rating: 5/5/5] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective 

Official PyTorch implementation of the paper ["Dataset Distillation with Neural Characteristic Function"](https://arxiv.org/abs/2502.20653) (NCFM) in CVPR 2025.


## :fire: News

- [2025/03/02] The code of our paper has been released.  
- [2025/02/27] Our NCFM paper has been accepted to CVPR 2025 (Rating: 555). Thanks!  


## :rocket: Pipeline

Here's an overview of the process behind our **Neural Characteristic Function Matching (NCFM)** method:

![Figure 1](./asset/figure1.png?raw=true)





## :mag: TODO

- [x] Distillation code
- [x] Evaluation code
- [ ] Config files
- [ ] Pretrained models
- [ ] Distilled datasets
- [ ] Continual learning code
- [ ] Project page




## 🛠️ Getting Started

To get started with NCFM, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/gszfwsb/NCFM.git
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```
3. Pretrain the models yourself, or download the **pretrained_models** from [Google Drive](https://drive.google.com/drive/folders/1HT_eUbTWOVXvBov5bM90b169jdy2puOh?usp=drive_link). 
```sh
cd pretrain
torchrun --nproc_per_node={n_gpus} --nnodes=1 pretrain_script.py --gpu={gpu_ids} --config_path=../config/{dataset}.yaml

```

4. Condense
```sh
cd condense 
torchrun --nproc_per_node={n_gpus} --nnodes=1 condense_script.py --gpu={gpu_ids} --ipc={ipc} --config_path=../config/{dataset}.yaml

```
5. Evaluation
```sh
cd evaluation 
torchrun --nproc_per_node={n_gpus} --nnodes=1 evaluation_script.py --gpu={gpu_ids} --ipc={ipc} --config_path=../config/{dataset}.yaml --load_path={distilled_dataset.pt}
```

### :blue_book: Example Usage

1. CIFAR-10

```sh
#ipc50
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=50 --config_path=../config/cifar10.yaml
```

2. CIFAR-100

```sh
#ipc10
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=10 --config_path=../config/cifar100.yaml
```



## :postbox: Contact
If you have any questions, please contact [Shaobo Wang](https://gszfwsb.github.io/)(`shaobowang1009@sjtu.edu.cn`).

## :pushpin: Citation
If you find NCFM useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{wang2025datasetdistillationneuralcharacteristic,
      title={Dataset Distillation with Neural Characteristic Function: A Minmax Perspective}, 
      author={Shaobo Wang and Yicun Yang and Zhiyuan Liu and Chenghao Sun and Xuming Hu and Conghui He and Linfeng Zhang},
      year={2025},
      eprint={2502.20653},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20653}, 
}
```

## Acknowledgement
We sincerely thank the developers of the following projects for their valuable contributions and inspiration: [MTT](https://github.com/GeorgeCazenavette/mtt-distillation), [DATM](https://github.com/NUS-HPC-AI-Lab/DATM), [DC/DM](https://github.com/VICO-UoE/DatasetCondensation), [IDC](https://github.com/VICO-UoE/DatasetCondensation), [SRe2L](https://github.com/VILA-Lab/SRe2L), [RDED](https://github.com/LINs-lab/RDED). We draw inspiration from these fantastic projects!
