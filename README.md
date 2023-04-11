# Best practices for 2-Body Pose Forecasting

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

The official PyTorch implementation of the **5th IEEE/CVF CVPR Precognition Workshop** paper [**Best Practices for 2-Body Pose Forecasting**](todo).

Visit our [**webpage**](https://www.pinlab.org/bestpractices2body) for more details.

![teaser](assets/teaser.png)

## Content
```
.
├── assets
│   ├── poses-viz.png
│   └── teaser.png
├── dataset
│   └── pi
│       └──
├── env.yaml
├── log
│   └──
├── README.md
├── snapshot
│   └── model-iter-40000.pth
├── src
│   ├── datasets
│   │   ├── data_utils_pi3d.py
│   │   ├── pi3d_hier.py
│   │   ├── pi3d.py
│   │   └── vis_2p.py
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   └── utils
│       ├── angle_to_joint.py
│       ├── config.py
│       ├── logger.py
│       ├── misc.py
│       ├── parser.py
│       ├── pyt_utils.py
│       ├── rigid_align.py
│       ├── util.py
│       └── visualize.py
└── viz
    └──
```
## Setup
### Environment
```
conda env create -f env.yaml
conda activate multi_body
```

### Dataset
Request ExPI dataset [here](https://team.inria.fr/robotlearn/multi-person-extreme-motion-prediction/) and place the `pi` folder under `datasets/`.

## Training
```
python train.py
```

## Test
```
python test.py
```

### Visualization
```
python test.py --visualize
```
## Results
![results](assets/poses-viz.png)

## TODO 
- Remove clutter.
- Add typing.