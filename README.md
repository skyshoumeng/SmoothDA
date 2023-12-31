# SmoothDA

<pre>
Leveraging Smooth Deformation Augmentation for LiDAR Point Cloud Semantic Segmentation
Shoumeng Qiu, Jie Chen, Chenghang Lai, Hong Lu, Xiangyang Xue, Jian Pu
Corresponding-author: Jian Pu
</pre>

### Overall Comparision

<p align="center">
        <img src="figs/comp.png" title="Comparision with other methods" width="70%">
</p> 

### Abstract

Existing data augmentation approaches on LiDAR point cloud are mostly developed on rigid transformation, such as rotation, flipping, or copy-based and mix-based methods, lacking the capability to generate diverse samples that depict smooth deformations in real-world scenarios. In response, we propose a novel and effective LiDAR point cloud augmentation approach with smooth deformations that can enrich the diversity of training data while keeping the topology of instances and scenes simultaneously. The whole augmentation pipeline can be separated into two different parts: scene augmentation and instance augmentation. To simplify the selection of deformation functions and ensure control over augmentation outcomes, we propose three effective strategies: residual mapping, space decoupling, and function periodization, respectively. We also propose an effective prior-based location sampling algorithm to paste instances on a more reasonable area in the scenes. Extensive experiments on both the SemanticKITTI and nuScenes challenging datasets demonstrate the effectiveness of our proposed approach across various baselines.


### Augmentation Pipeline

<p align="center">
        <img src="figs/pipeline.png" title="Augmentation Pipeline" width="90%">
</p>

### Instances augmentation samples

<p align="center">
        <img src="figs/insts.png" title="Instaces augmentation results" width="70%">
</p>

### Scenes augmentation samples

<p align="center">
        <img src="figs/scenes.png" title="Scenes augmentation results" width="90%">
</p>

### Code 

Our code borrows heavily from [PolarMix](https://github.com/xiaoaoran/polarmix). For the training details, please refer to the instructions provided in the PolarMix project.

#### Training on SemanticKITTI

Please generate the instances database first, following the instructions provided in [Panoptic-PolarNet](https://github.com/edwardzhou130/Panoptic-PolarNet).


Then you may run the following code to train the model from scratch. 

SPVCNN:
```bash
python train.py configs/semantic_kitti/spvcnn/cr0p5.yaml --run-dir runs/semantickitti/spvcnn_polarmix --distributed False
```
MinkowskiNet:
```bash
python train.py configs/semantic_kitti/minkunet/cr0p5.yaml --run-dir run/semantickitti/minkunet_polarmix --distributed False
```

### Thanks
We thank the opensource project [PolarMix](https://github.com/xiaoaoran/polarmix).

