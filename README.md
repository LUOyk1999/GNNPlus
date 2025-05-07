# 🔍 Research Series on Classic GNNs

| Benchmarking Series: Reassessing Classic GNNs | Paper |
| - | - |
| [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://github.com/LUOyk1999/tunedGNN) (NeurIPS 2024) | [Link](https://openreview.net/forum?id=xkljKdGe4E) |
| **_[GNN+: Can Classic GNNs Be Strong Baselines for Graph-Level Tasks?](https://github.com/LUOyk1999/GNNPlus) (ICML 2025)_** | [Link](https://arxiv.org/abs/2502.09263) | 

| Follow-up Studies | Paper |
| - | - |
| [When Dropout Meets Graph Convolutional Networks](https://github.com/LUOyk1999/dropout-theory) (ICLR 2025)  | [Link](https://openreview.net/forum?id=PwxYoMvmvy) | 
| [Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning](https://github.com/LUOyk1999/NodeID) (ICLR 2025) | [Link](https://openreview.net/forum?id=t9lS1lX9FQ) | 

# GNN+: Can Classic GNNs be Strong Baselines for Graph-level Tasks? (ICML 2025)
[![arXiv](https://img.shields.io/badge/arXiv-2502.09263-b31b1b.svg)](https://arxiv.org/abs/2502.09263) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unlocking-the-potential-of-classic-gnns-for/graph-property-prediction-on-ogbg-ppa)](https://paperswithcode.com/sota/graph-property-prediction-on-ogbg-ppa?p=unlocking-the-potential-of-classic-gnns-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unlocking-the-potential-of-classic-gnns-for/graph-classification-on-malnet-tiny)](https://paperswithcode.com/sota/graph-classification-on-malnet-tiny?p=unlocking-the-potential-of-classic-gnns-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unlocking-the-potential-of-classic-gnns-for/graph-classification-on-cifar10-100k)](https://paperswithcode.com/sota/graph-classification-on-cifar10-100k?p=unlocking-the-potential-of-classic-gnns-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unlocking-the-potential-of-classic-gnns-for/graph-regression-on-peptides-struct)](https://paperswithcode.com/sota/graph-regression-on-peptides-struct?p=unlocking-the-potential-of-classic-gnns-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unlocking-the-potential-of-classic-gnns-for/node-classification-on-cluster)](https://paperswithcode.com/sota/node-classification-on-cluster?p=unlocking-the-potential-of-classic-gnns-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unlocking-the-potential-of-classic-gnns-for/node-classification-on-coco-sp)](https://paperswithcode.com/sota/node-classification-on-coco-sp?p=unlocking-the-potential-of-classic-gnns-for)

Based on the GPS codebase: https://github.com/rampasek/GraphGPS

### Python environment setup with Conda

Tested with Python 3.9/3.10, PyTorch 2.2.0, and PyTorch Geometric 2.3.1.

To set up the environment, run the following commands:
```bash
conda create -n GNNPlus python=3.10
conda activate GNNPlus

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install scikit-learn==1.4.0
pip install fsspec rdkit
pip install pytorch-lightning yacs torchmetrics
pip install networkx
pip install tensorboardX
pip install ogb
pip install wandb
```


### Running Training

To execute training, activate the environment and run the following commands:

```bash
conda activate GNNPlus

sh run.sh 0 cifar10 2 > cifar10.txt 2>&1 &
sh run.sh 1 cluster 2 > cluster.txt 2>&1 &
sh run.sh 2 coco 2 > coco.txt 2>&1 &
sh run.sh 3 code2 1 > code2.txt 2>&1 &
sh run.sh 4 hiv 2 > hiv.txt 2>&1 &
sh run.sh 5 mal 5 > mal.txt 2>&1 &
sh run.sh 6 zinc 2 > zinc.txt 2>&1 &
sh run.sh 7 pattern 4 > pattern.txt 2>&1 &

sh run.sh 2 pcba 2 > pcba.txt 2>&1 &
sh run.sh 3 peptides-func 4 > peptides-func.txt 2>&1 &
sh run.sh 4 peptides-struct 4 > peptides-struct.txt 2>&1 &
sh run.sh 5 voc 2 > voc.txt 2>&1 &
sh run.sh 6 ppa 2 > ppa.txt 2>&1 &
sh run.sh 7 mnist 2 > mnist.txt 2>&1 &
```

Alternatively, use the following format for executing training runs:

```bash
conda activate GNNPlus

python main.py --cfg configs/gcn/peptides-func.yaml --repeat 2 seed 0

python main.py --cfg configs/gatedgcn/ppa.yaml --repeat 2 seed 0 
```

## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2025unlocking,
  title={Unlocking the Potential of Classic GNNs for Graph-level Tasks: Simple Architectures Meet Excellence},
  author={Luo, Yuankai and Shi, Lei and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2502.09263},
  year={2025}
}
```
