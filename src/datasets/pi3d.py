"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
MIT license.
"""
# pi3d.py

import numpy as np

import torch
from torch.utils.data import Dataset

from src.datasets import data_utils_pi3d
from src.utils.config import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Datasets(Dataset):
    def __init__(self, opt, actions=None, is_train=True):
        self.path_to_data = opt.data_dir + opt.pi3d_anno_dir
        self.is_train = is_train
        if is_train:  # train
            self.in_n = opt.motion.pi3d_input_length
            self.out_n = opt.motion.pi3d_target_length_train
            self.split = 0
        else:  # test
            self.in_n = opt.motion.pi3d_input_length
            self.out_n = opt.motion.pi3d_target_length_eval
            self.split = 1
        self.skip_rate = 1
        self.p3d = {}
        sampled_seq = []

        if opt.protocol == "pro3":  # unseen action split
            if is_train:  # train on acro2
                acts = [
                    "2/a-frame",
                    "2/around-the-back",
                    "2/coochie",
                    "2/frog-classic",
                    "2/noser",
                    "2/toss-out",
                    "2/cartwheel",
                    "1/a-frame",
                    "1/around-the-back",
                    "1/coochie",
                    "1/frog-classic",
                    "1/noser",
                    "1/toss-out",
                    "1/cartwheel",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 4, 5, 6],
                    [1, 2, 3, 4, 6],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                ]

            else:  # test on acro1
                # ! In the paper they have a different order.
                acts = [
                    "2/crunch-toast",
                    "2/frog-kick",
                    "2/ninja-kick",
                    "1/back-flip",
                    "1/big-ben",
                    "1/chandelle",
                    "1/check-the-change",
                    "1/frog-turn",
                    "1/twisted-toss",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 4, 5, 8],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                ]

                if (
                    opt.test_split is not None
                ):  # test per action for unseen action split
                    acts, subfix = [acts[opt.test_split]], [subfix[opt.test_split]]

        else:  # common action split and single action split
            if is_train:  # train on acro2
                acts = [
                    "2/a-frame",
                    "2/around-the-back",
                    "2/coochie",
                    "2/frog-classic",
                    "2/noser",
                    "2/toss-out",
                    "2/cartwheel",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                ]

                if opt.protocol in [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                ]:  # train per action for single action split
                    acts = [acts[int(opt.protocol)]]
                    subfix = [subfix[int(opt.protocol)]]

            else:  # test on acro1
                acts = [
                    "1/a-frame",
                    "1/around-the-back",
                    "1/coochie",
                    "1/frog-classic",
                    "1/noser",
                    "1/toss-out",
                    "1/cartwheel",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 4, 5, 6],
                    [1, 2, 3, 4, 6],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                ]

                if (
                    opt.test_split is not None
                ):  # test per action for common action split
                    acts, subfix = [acts[opt.test_split]], [subfix[opt.test_split]]
                if opt.protocol in [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                ]:  # test per action for single action split
                    acts, subfix = [acts[int(opt.protocol)]], [
                        subfix[int(opt.protocol)]
                    ]

        for action_idx in np.arange(len(acts)):
            subj_action = acts[action_idx]
            subj, action = subj_action.split("/")
            for subact_i in np.arange(len(subfix[action_idx])):
                subact = subfix[action_idx][subact_i]
                print(
                    "Reading subject {0}, action {1}, subaction {2}".format(
                        subj, action, subact
                    )
                )
                filename = "{0}/acro{1}/{2}{3}/mocap_cleaned.tsv".format(
                    self.path_to_data, subj, action, subact
                )
                the_sequence = data_utils_pi3d.readCSVasFloat(filename, with_key=True)
                num_frames = the_sequence.shape[0]
                the_sequence = data_utils_pi3d.normExPI_2p_by_frame(the_sequence)
                the_sequence = torch.from_numpy(the_sequence).float().to(device)

                if self.is_train:  # train
                    seq_len = self.in_n + self.out_n
                    valid_frames = np.arange(
                        0, num_frames - seq_len + 1, self.skip_rate
                    )
                else:  # test
                    seq_len = self.in_n + 30
                    valid_frames = data_utils_pi3d.find_indices_64(num_frames, seq_len)

                p3d = the_sequence
                the_sequence = p3d.view(num_frames, -1).cpu().data.numpy()
                fs_sel = valid_frames
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, valid_frames + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]

                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)

        self.dimension_use = np.arange(18 * 2 * 3)
        self.in_features = len(self.dimension_use)
        self.gts = sampled_seq

    def __len__(self):
        return self.gts.shape[0]

    def __getitem__(self, item):
        gts = {}
        gts = self.gts[item]
        pi3d_motion_input = gts[: self.in_n] / 1000  # meter
        pi3d_motion_target = gts[self.in_n :] / 1000  # meter
        return pi3d_motion_input, pi3d_motion_target
