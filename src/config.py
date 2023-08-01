# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C


# please config ROOT_dir and user when u first using
C.abs_dir = os.getcwd()
# C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
# C.repo_name = "unet"
C.repo_name = C.this_dir
C.root_dir = C.abs_dir[: C.abs_dir.index(C.repo_name) + len(C.repo_name)]
# C.root_dir = C.abs_dir[: C.abs_dir.index(C.repo_name))]


C.log_dir = osp.abspath(osp.join(C.abs_dir, "log"))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))


exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_dir + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

# Dataset path
C.data_dir = "./dataset/"


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, "lib"))

# Dataset Config
C.pi3d_anno_dir = "pi"


C.motion = edict()

# Input and output time frame.
C.motion.pi3d_input_length = 10  # 10
# DCT size.
C.motion.pi3d_input_length_dct = 10  # 10
# Kernel size at test time.
C.motion.pi3d_target_length_train = 10  # 10
# Eval output time frame.
C.motion.pi3d_target_length_eval = 25
C.motion.dim = 108

# 'pro1: common action split;0-6: single action split;pro3: unseen action split')
# ! Choose accordingly.
C.protocol = "pro1"
C.test_split = None  # help for test protocol1

C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = False
C.use_curriculum_loss = False
# Models
C.unet_inpalightpa = True

#  Model Config
## Network
# C.pre_dct = False
# C.post_dct = False
## Motion Network mlp
# dim_ = 108

# Train Config
C.batch_size = 2
C.batch_size_test = 4
C.num_workers = 2  # 8

C.cos_lr_max = 1e-5
C.cos_lr_min = 5e-8
C.cos_lr_total_iters = 40000

C.weight_decay = 1e-4
C.model_pth = None

# Eval Config
C.shift_step = 1

# Display Config
C.print_every = 100
C.save_every = 5000


if __name__ == "__main__":
    # print all config
    print(C)

    # print snapshot dir
    print(C.snapshot_dir)

    print(config.motion)
