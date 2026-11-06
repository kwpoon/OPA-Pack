# OPA-Pack: Object-Property-Aware Robotic Bin Packing

Jia-Hui Pan, Yeok Tatt Cheah, Zhengzhe Liu, Ka-Hei Hui, Xiaojie Gao, Pheng-Ann Heng, Yun-Hui Liu, and Chi-Wing Fu

**[codes coming soon] · [dataset](DATASET_URL)**

![OPA-Pack framework: object property recognition and property-aware packing learning.](assets/framework.png)

OPA-Pack combines object-property recognition with reinforcement learning to pack everyday objects compactly while separating incompatible items and reducing loads on fragile objects.

## Method

**1. Recognize object properties.** A vision-language model takes four rendered views, the category, and the size of each object to predict eleven physical and semantic properties. CLIP-based retrieval supplies similar annotated examples, while material-aware reasoning estimates inner and outer materials to infer fragility and density. Rules then identify incompatible pairs, including sharp and soft objects, chemicals and food, medicines and food, and ignition and flammable items.

**2. Represent objects and the container.** OPA-Net combines PointNet geometry features with learned object-property embeddings. Three heightmaps describe the container: occupancy records the packed surface, fragility tracks fragile objects, and a candidate-specific avoidance map locates packed objects incompatible with the object being considered.

**3. Learn what to pack and where.** An object Q-predictor selects the next object from the buffer. A placement Q-predictor then scores candidate orientations and positions to choose its placement. OPA-Net is trained with dueling deep Q-learning and a reward that jointly encourages compact packing, reduces load on fragile objects, and separates incompatible pairs.

## Dataset and Results

- **Dataset:** 1,032 everyday objects with physical and semantic property annotations.
- **Simulation:** 95.0% avoidance accuracy and 29.4% lower load on fragile objects, with comparable packing compactness to the baseline.
- **Real-world validation:** 70 packing sequences sampled from 100 scanned objects.
