# Trident: A Provider-Oriented Resource Management
Framework for Serverless Computing Platforms

This repository contains the core code of paper: Trident: A Provider-Oriented Resource Management Framework for Serverless Computing Platforms.

## Introduction

Trident is a provider-oriented resource management framework for serverless computing platforms. It introduces:

- A novel dynamic model selection algorithm for more accurate workload prediction. 
- A hierarchical reinforcement learning (HRL)-based approach for VM provisioning with a mix of types and configurations. 
- An effective collocation placement strategy for efficient function container scheduling.

##### env

This directory contains the serverless simulation environment implemented by Open AI `Gym` . To use it, please download the serverless dataset from https://github.com/Azure/AzurePublicDataset and put the dataset into this directory.

##### functionplacer

This directory contains the implementation of collocation function placement module of Trident.

##### modelselector

This directory contains the implementation of dynamic model selection module of Trident.

##### provisioner

This directory contains the implementation of HRL-based VM provisioner implemented by `torch`.

## Installation

```
pip install -r requirements.txt
```

## Citation format

B. Zhu, Y. Zhu, C. Chen, L. Kong. "Trident: A Provider-Oriented Resource Management Framework for Serverless Computing Platforms", in *IEEE Transactions on Services Computing (TSC)*, 2025

```
@article{zhu2025,
  title={Trident: A Provider-Oriented Resource Management Framework for Serverless Computing Platforms},
  author={Zhu, Botao and Zhu, Yifei and Chen, Chen and Kong, Linghe},
  journal={IEEE Transactions on Services Computing (TSC)},
  year={2025}
}
```

