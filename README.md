<h1 align="center">OPA-Pack: Object-Property-Aware Robotic Bin Packing</h1>

<p align="center">
  Jia-Hui Pan, Yeok Tatt Cheah, Zhengzhe Liu, Ka-Hei Hui, Xiaojie Gao,
</p>
<p align="center">
Pheng-Ann Heng, Yun-Hui Liu, and Chi-Wing Fu
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.13339"><img alt="Paper" src="https://img.shields.io/static/v1?label=Paper&amp;message=arXiv%3A2505.13339&amp;color=b31b1b&amp;logo=arxiv"></a>
  &ensp;
  <a href="https://github.com/kwpoon/OPA-Pack/"><img alt="Project repository" src="https://img.shields.io/static/v1?label=Project&amp;message=GitHub&amp;color=blue&amp;logo=github"></a>
  &ensp;
  <a href="https://drive.google.com/file/d/1iTPi-63smbNX3Aqw7QbGNJOHnUY49utO/view?usp=sharing"><img alt="OPA dataset" src="https://img.shields.io/static/v1?label=OPA&amp;message=Google%20Drive&amp;color=green&amp;logo=googledrive"></a>
</p>

---

OPA-Pack combines object-property information with reinforcement learning to pack irregular objects compactly, separate incompatible items, and reduce loads on fragile objects. This repository contains the packing-policy training and simulation evaluation code.

## 🧠 Method

1. **Object properties.** The method uses multi-view vision-language reasoning and retrieved examples to predict physical and semantic properties. Rules identify incompatible object pairs. The packing code consumes the prepared property annotations and pair lists in the dataset; training does not invoke a vision-language model.
2. **Geometry and property representation.** OPA-Net combines point-cloud geometry with object-property features. Container heightmaps describe occupancy, fragility, and avoidance regions for the object being considered.
3. **Buffered packing.** An order Q-network chooses an object from the buffer, and a location Q-network chooses its placement. The hierarchical policy learns from volume utilization, incompatibility, and fragile-object penalties.

The command-line flag is spelled `--hierachical` in the code; use that spelling in commands.

## 🛠️ Installation

The supplied environment targets Linux x86_64 and an NVIDIA GPU with a driver supporting the PyTorch CUDA 12.8 build.

```bash
conda env create -f env_opa_pack.yml
conda activate opa_pack
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

[env_opa_pack.yml](env_opa_pack.yml) specifies Python 3.11, PyTorch 2.7.0 + CUDA 12.8, NumPy, Gym, trimesh, PyBullet, and the other runtime packages.

## 📦 Dataset

Download the [processed OPA dataset](https://drive.google.com/file/d/1iTPi-63smbNX3Aqw7QbGNJOHnUY49utO/view?usp=sharing) and place the subset directories under `dataset/`.

| Subset | Command-line name | Objects |
| --- | --- | ---: |
| Daily Necessities | `oop_daily` | 233 |
| Hardware and Stationery | `oop_tool` | 302 |
| Kitchen | `oop_cook` | 497 |
| **Total** | | **1,032** |

These are object counts from the dataset description; IDs in `id2shape.pt` can represent different prepared poses and need not have the same count.

Each object has eleven predicted properties: medicine, edible item, household chemical, ignition item, flammable item, softness, sharpness, inner material, outer material, fragility, and density level.

Expected layout, illustrated with daily:

```text
dataset/
├── oop_daily/
│   ├── shape_vhacd/                       # Meshes for physics simulation
│   ├── pointCloud/                        # Prepared point clouds
│   ├── shape_info_json/                   # Object-property annotations
│   ├── id2shape.pt                        # Prepared shape/pose ID mapping
│   ├── test_sequence.pt                   # Fixed evaluation sequences
│   └── dict_obj_forbid_list_oop_daily.json # Incompatible-object lists
├── oop_tool/
└── oop_cook/
```

The other subsets use `dict_obj_forbid_list_oop_tool.json` and `dict_obj_forbid_list_oop_cook.json`. The runtime reads `shape_info_json` and these JSON pair lists.

## 🚀 Training

Activate the Conda environment, then use the script for the desired subset:

```bash
bash train_daily.sh
bash train_tool.sh
bash train_cook.sh
```

Each command runs one subset in the foreground. Use separate terminals to run subsets concurrently, subject to available memory. Each script creates a timestamped log directory and automatically evaluates saved checkpoints on 200 test sequences.

| Setting | Value |
| --- | ---: |
| Parallel environments | 16 |
| Batch size | 64 |
| T-max | 100,000 |
| Object buffer size | 10 |
| Sampled points | 1,024 |
| Placement candidates | 500 |
| Warmup steps | 500 |
| Replay capacity | 100,000 |
| Volume reward weight | 10 |
| Alpha / beta | 0.2 / 20 |
| Checkpoint interval | 5,000 learner updates |
| Additional checkpoint | 1,000 learner updates |
| Evaluation sequences per checkpoint | 200 |

An optional first argument can be used to specify an output directory:

```bash
bash train_daily.sh ./logs/daily_run
```

Scripts: [daily](train_daily.sh), [tool](train_tool.sh), [cook](train_cook.sh). Full training and evaluation commands are recorded in the corresponding log subdirectories.

### Saving checkpoints

- **`--save-interval 5000`** saves paired checkpoints at learner updates 5,000, 10,000, and so on in this distributed hierarchical path. `--checkpoint-interval` is a legacy setting and does not control that schedule.

The replay capacity is divided across environments separately for the order and location memories. The sampler divides a batch across those memories; use a batch size divisible by the environment count, such as 64 with 16 environments.

### Continue from saved weights

Pass both checkpoint files and the completed learner-update count, together with the same model and environment settings:

```bash
bash run_opa_pack.sh \
  --device 0 --dataset oop_daily --custom daily_continue \
  --hierachical --bufferSize 10 --distributed \
  --num_processes 16 --batch-size 64 \
  --samplePointsNum 1024 --selectedAction 500 \
  --resolutionA 0.02 --resolutionH 0.01 \
  --volume-weight 10 --alpha 0.2 --beta 20 \
  --ordmodel ./logs/daily_run/checkpoints/update_00025000/order.pt \
  --locmodel ./logs/daily_run/checkpoints/update_00025000/location.pt \
  --start-update 25000 --T-max 100000 --save-interval 5000 \
  --output-dir ./logs/daily_continue
```

These checkpoints contain network weights, not a full training-state snapshot. Optimizer state and replay buffers start fresh, and warmup runs again. `--start-update` preserves the update numbering; `T-max` specifies the new run's additional environment steps.

### Output layout

```text
logs/daily_tool_run/
├── experiment.json
├── comparison.csv                 # All checkpoint rounds and evaluation states
├── comparison.html
├── latest_comparison.csv          # Latest round fully evaluated for all selected subsets
├── latest_comparison.html
├── run_all.sh                     # Training commands only; does not launch the evaluator
├── oop_daily/
│   ├── run.sh
│   ├── parameters.json
│   ├── train.log
│   ├── actor_progress.json
│   ├── progress.json
│   ├── source/
│   ├── tensorboard/
│   ├── checkpoints/
│   │   ├── latest.json
│   │   └── update_00005000/
│   │       ├── order.pt
│   │       ├── location.pt
│   │       └── ready.json
│   └── evaluation/update_00005000/
│       ├── run.sh
│       ├── parameters.json
│       ├── status.json
│       ├── evaluation.log
│       └── metrics.json
└── oop_tool/
```

`ready.json` is published after both network files have been saved. Evaluation folders also contain trajectory files. The latest shared table waits for all selected subsets; it does not mix different checkpoint rounds. If one subset fails before a round is saved, the others' checkpoints remain available, but that shared round waits for the missing subset.

## 📊 Evaluate a checkpoint manually

Evaluate both networks from the same checkpoint directory. Match the training configuration, including the reward weights when comparing reward values.

```bash
bash run_opa_pack.sh \
  --evaluate --device 0 --dataset oop_daily --custom daily_eval \
  --hierachical --bufferSize 10 \
  --samplePointsNum 1024 --selectedAction 500 \
  --resolutionA 0.02 --resolutionH 0.01 \
  --volume-weight 10 --alpha 0.2 --beta 20 \
  --seed 123 --evaluation-episodes-test 100 \
  --ordmodel ./logs/daily_run/checkpoints/update_00025000/order.pt \
  --locmodel ./logs/daily_run/checkpoints/update_00025000/location.pt \
  --output-dir ./logs/daily_run/evaluation/update_00025000
```

Evaluation reads `dataset/<subset>/test_sequence.pt`, starting from the first sequence. Use the same sequence file, seed, episode count, and environment settings for comparisons. The evaluator uses one environment. Direct `main.py` evaluation writes `metrics.json` and trajectories; the experiment launcher additionally records the command, status, and redirected console log.

## 📐 Geometry and resource settings

The configured bin is **0.32 × 0.32 × 0.30 m**. With the example resolutions:

- `resolutionH=0.01`: a heightmap cell covers a 1 cm × 1 cm horizontal area.
- `resolutionA=0.02`: candidate horizontal placement locations are sampled at 2 cm intervals.
- `resolutionZ=0.01` and `resolutionRot=8`: vertical resolution and rotation settings.

Changing the bin size or heightmap dimensions may require corresponding model-input changes.

Parallel environments, replay buffers, and dataset loading consume **system RAM**, independently of GPU memory and disk storage. Measure the container's memory limit, not just the host's reported total memory. More environments do not always improve throughput. Concurrent evaluation adds resource use, and replay memory grows during training. Select concurrency and replay capacity based on measured peak usage; 16 environments is not a guarantee against memory-related interruption.

A paired checkpoint is approximately 3.84 MiB in the current architecture. Logs and evaluation trajectories consume additional disk space. `train.log`, `actor_progress.json`, and `progress.json` help distinguish active environment collection from learner updates; a changing learner counter alone does not prove that new experience is still being collected.

## 🗂️ Code guide

| File or directory | Role |
| --- | --- |
| `main.py`, `arguments.py` | Entry point and configuration |
| `model.py`, `agent.py`, `memory.py` | Networks, learning, and replay memory |
| `trainer.py` | Environment collection and asynchronous learner |
| `environment/physics0/` | Packing simulation, reward, sequence loading, and metrics |
| `wrapper/` | Vectorized environments and process communication |
| `tools.py` | Data preparation/loading and evaluation routines |
| `env_opa_pack.yml`| Runtime environment |
| `run_opa_pack.sh` | entry-point wrapper |

## 📝 Acknowledgments and citation

Please cite the [OPA-Pack paper](https://arxiv.org/abs/2505.13339) when using this work:
```bibtex
@article{pan2026opa,
  title={OPA-Pack: Object-Property-Aware Robotic Bin Packing},
  author={Pan, Jia-Hui and Cheah, Yeok Tatt and Liu, Zhengzhe and Hui, Ka-Hei and Gao, Xiaojie and Heng, Pheng-Ann and Liu, Yun-Hui and Fu, Chi-Wing},
  journal={IEEE Transactions on Robotics},
  year={2026},
  publisher={IEEE}
}
```


The packing implementation builds on IR-BPP; please also acknowledge its contribution:

```bibtex
@article{zhao2022learning,
  title={Learning Physically Realizable Skills for Online Packing of General 3D Shapes},
  author={Zhao, Hang and Pan, Zherong and Yu, Yang and Xu, Kai},
  journal={ACM Transactions on Graphics},
  year={2022},
  publisher={ACM New York, NY}
}
```

## 📄 Usage terms

> This source code is released only for academic use. Please do not use it for commercial purposes without authorization of the author.
