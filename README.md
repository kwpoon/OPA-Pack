<h1 align="center">
  OPA-Pack: Object-Property-Aware Robotic Bin Packing
</h1>

<div align="center">
Jia-Hui Pan<sup>1</sup>, Yeok Tatt Cheah<sup>2</sup>, Zhengzhe Liu<sup>3</sup>, Ka-Hei Hui<sup>1</sup>,<br>
Xiaojie Gao<sup>1</sup>, Pheng-Ann Heng<sup>1</sup>, Yun-Hui Liu<sup>4</sup>, and Chi-Wing Fu<sup>1</sup>
</div>

<br>

<div align="center">
<sup>1</sup>Department of Computer Science and Engineering, The Chinese University of Hong Kong<br>
<sup>2</sup>Hong Kong Centre for Logistics Robotics<br>
<sup>3</sup>School of Data Science, Lingnan University<br>
<sup>4</sup>Department of Mechanical and Automation Engineering, The Chinese University of Hong Kong
</div>

<br>

<div align="center">
<strong>Accepted for publication in IEEE Transactions on Robotics (T-RO), 2026.</strong>
</div>

<br>

<div align="center">
  <a href="https://arxiv.org/abs/2505.13339"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper on arXiv"></a>
  &ensp;
  <a href="#dataset"><img src="https://img.shields.io/badge/Dataset-OPA-yellow" alt="OPA dataset description"></a>
  &ensp;
  <a href="#method"><img src="https://img.shields.io/badge/Method-OPA--Net-blue" alt="OPA-Net method"></a>
</div>

---

This repository accompanies our paper **OPA-Pack: Object-Property-Aware Robotic Bin Packing**. OPA-Pack incorporates object properties into robotic bin-packing planning to reduce load on fragile objects and separate incompatible object pairs while preserving packing compactness.

Our framework consists of two stages. First, an object-property recognition scheme combines a vision-language model with retrieval-augmented generation and chain-of-thought reasoning to predict object-centric properties and derive pairwise avoidance relations. This process produces the **OPA dataset**, containing property annotations for **1,032 everyday objects**. Second, **OPA-Net** encodes candidate-object geometry and properties, together with occupancy, fragility, and avoidance heightmaps, to select the next object and its placement. The network is trained using deep Q-learning with a reward that jointly considers packing compactness, load on fragile objects, and separation of incompatible pairs.

<p align="center"><a href="figures/oop_pack_framework_v9.pdf"><img src="figures/oop_pack_framework_v9.pdf" alt="OPA-Pack framework" width="100%"></a></p>

## 🛠️ Environment Setup

The virtual packing experiments use **PyBullet**. Object-property recognition used **GPT-4o** and **CLIP**.

<a id="dataset"></a>

## 📦 Dataset Preparation

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

<a id="method"></a>

## 🧩 Key Implementation

OPA-Pack includes the following main components:

- **Object-property recognition:** predict object properties from four orthogonal object views, the object class name, and size information using GPT-4o, CLIP-based retrieval, and material-guided chain-of-thought reasoning.
- **OPA-Net:** encode object point clouds and properties, and predict Q-values for object selection and placement selection using a dueling DQN architecture.
- **Packing reward and learning:** jointly optimize packing compactness, load on fragile objects, and separation of incompatible object pairs.
- **Sequential packing:** select an object from the buffer, evaluate candidate placements, and update the packing state after each placement.

## 📊 Evaluation Results

In the virtual experiments, OPA-Pack improves avoidance accuracy from **52.3% to 95.0%** and reduces load on fragile objects by **29.4%**, while preserving packing compactness.

| Metric | Baseline | OPA-Pack |
| --- | ---: | ---: |
| Avoidance accuracy ↑ | 52.3% | **95.0%** |
| Packing compactness ↑ | 0.420 | 0.413 |

We also evaluate OPA-Pack on a physical platform using **70 packing sequences sampled from 100 scanned real-world objects**.

## 🔗 Citation

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
