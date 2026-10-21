## OPA-Pack: Object-Property-Aware Robotic Bin Packing

#### Jia-Hui Pan<sup>1</sup>, Yeok Tatt Cheah<sup>2</sup>, Zhengzhe Liu<sup>3</sup>, Ka-Hei Hui<sup>1</sup>, Xiaojie Gao<sup>1</sup>, Pheng-Ann Heng<sup>1</sup>, Yun-Hui Liu<sup>4</sup>, and Chi-Wing Fu<sup>1</sup>

<sup>1</sup>Department of Computer Science and Engineering, The Chinese University of Hong Kong  
<sup>2</sup>Hong Kong Centre for Logistics Robotics  
<sup>3</sup>School of Data Science, Lingnan University  
<sup>4</sup>Department of Mechanical and Automation Engineering, The Chinese University of Hong Kong

**Accepted for publication in IEEE Transactions on Robotics (T-RO), 2026.**

[Paper (arXiv)](https://arxiv.org/abs/2505.13339)

### Introduction

This repository accompanies our paper **OPA-Pack: Object-Property-Aware Robotic Bin Packing**. OPA-Pack incorporates object properties into robotic bin-packing planning to reduce load on fragile objects and separate incompatible object pairs while preserving packing compactness.

Our framework consists of two stages. First, an object-property recognition scheme combines a vision-language model with retrieval-augmented generation and chain-of-thought reasoning to predict object-centric properties and derive pairwise avoidance relations. This process produces the **OPA dataset**, containing property annotations for **1,032 everyday objects**. Second, **OPA-Net** encodes candidate-object geometry and properties, together with occupancy, fragility, and avoidance heightmaps, to select the next object and its placement. The network is trained using deep Q-learning with a reward that jointly considers packing compactness, load on fragile objects, and separation of incompatible pairs.

In the virtual experiments, OPA-Pack improves avoidance accuracy from **52.3% to 95.0%**, reduces load on fragile objects by **29.4%**, and achieves a packing compactness of **0.413**, compared with **0.420** for the baseline. We also evaluate OPA-Pack on a physical platform using **70 packing sequences sampled from 100 scanned real-world objects**.

### Dataset Preparation

The OPA dataset contains **1,032 objects** collected from **Objaverse-v1**, divided into three subsets:

| Subset | Number of objects |
| --- | ---: |
| Kitchen | 497 |
| Hardware and Stationery | 302 |
| Daily Necessities | 233 |
| **Total** | **1,032** |

For each object, the dataset retains its 3D geometry and class name and includes eleven predicted object-centric properties:

- **Semantic properties:** medicine, edibility, household chemical, ignition item, and flammable item.
- **Physical properties:** softness, sharpness, inner material, outer material, fragility, and density level.

Pairwise avoidance relations are derived from these properties for four types of incompatible pairs: sharp objects and soft objects; medicines and edible objects; household chemicals and edible objects; and ignition items and flammable items.

The property-recognition pipeline uses an auxiliary retrieval database of **40 manually annotated YCB objects**. For each query object, CLIP retrieves the ten most similar examples to support property prediction.

**TODO:** Add the dataset download link, released directory structure, and extraction instructions.

### Key Implementation

OPA-Pack includes the following main components:

- **Object-property recognition:** predict object properties from four orthogonal object views, the object class name, and size information using GPT-4o, CLIP-based retrieval, and material-guided chain-of-thought reasoning.
- **OPA-Net:** encode object point clouds and properties, and predict Q-values for object selection and placement selection using a dueling DQN architecture.
- **Packing reward and learning:** jointly optimize packing compactness, load on fragile objects, and separation of incompatible object pairs.
- **Sequential packing:** select an object from the buffer, evaluate candidate placements, and update the packing state after each placement.

### Installation

The virtual packing experiments use **PyBullet**. Object-property recognition used **GPT-4o** and **CLIP**.


### Citation

If you find this work useful, please cite:

```bibtex
@article{pan2026opapack,
  title   = {{OPA-Pack}: Object-Property-Aware Robotic Bin Packing},
  author  = {Pan, Jia-Hui and Cheah, Yeok Tatt and Liu, Zhengzhe and Hui, Ka-Hei and Gao, Xiaojie and Heng, Pheng-Ann and Liu, Yun-Hui and Fu, Chi-Wing},
  journal = {IEEE Transactions on Robotics},
  year    = {2026},
  note    = {Accepted for publication}
}
```

<!-- Add the DOI, volume, issue, and page numbers when the final bibliographic record is available. -->
