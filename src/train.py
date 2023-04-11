from typing import Tuple
import json
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from datasets import pi3d as datasets
from src.utils.config import config
from src.model import Model
from src.utils.logger import get_logger, print_and_log_info
from src.utils.pyt_utils import link_file, ensure_dir
from src.utils.util import *
from src.utils.parser import Parser

from test import test


def train_step(
    pi3d_motion_input: torch.Tensor,
    pi3d_motion_target: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    nb_iter: int,
    total_iter: int,
    max_lr: float,
    min_lr: float,
) -> Tuple[float, torch.optim.Optimizer, float]:
    """Train step.

    Args:
        pi3d_motion_input (torch.Tensor): Input sequence.
        pi3d_motion_target (torch.Tensor): Ground truth sequence.
        model (nn.Module): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
        nb_iter (int): Iteration number.
        total_iter (int): Total number of iterations.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        Tuple[float, torch.optim.Optimizer, float]: Loss, optimizer, learning rate.
    """
    # input b,50,66 output b,10,66
    if config.deriv_input:
        b, n, c = pi3d_motion_input.shape
        pi3d_motion_input_ = pi3d_motion_input.clone()
        pi3d_motion_input_ = torch.matmul(
            dct_m[:, :, : config.motion.pi3d_input_length],
            pi3d_motion_input_.to(device),
        )
    else:
        pi3d_motion_input_ = pi3d_motion_input.clone()

    motion_pred = model(pi3d_motion_input_.to(device))  # in out is 2,10,108

    motion_pred = torch.matmul(
        idct_m[:, : config.motion.pi3d_input_length, : config.motion.pi3d_input_length],
        motion_pred,
    )

    # always predict the same size as the input, when dealing with loss we cut to our actual output size.
    # no need to redo previous trainings since we used 10 and 10.
    motion_pred = motion_pred[:, : config.motion.pi3d_target_length_train, :]

    if config.deriv_output:
        offset = pi3d_motion_input[:, -1:].to(device)  # 2,1,66
        motion_pred = (
            motion_pred[:, : config.motion.pi3d_target_length] + offset
        )  # 2,10,66
    else:
        motion_pred = motion_pred[:, : config.motion.pi3d_target_length]

    loss_l = torch.mean(
        torch.norm((motion_pred - pi3d_motion_target.to(device))[:, :, :54], dim=2)
    )
    loss_f = torch.mean(
        torch.norm((motion_pred - pi3d_motion_target.to(device))[:, :, 54:], dim=2)
    )
    b, n, c = pi3d_motion_target.shape
    motion_pred = motion_pred.reshape(b, n, 36, 3).reshape(-1, 3)
    pi3d_motion_target = (
        pi3d_motion_target.to(device).reshape(b, n, 36, 3).reshape(-1, 3)
    )
    loss = torch.mean(torch.norm(motion_pred - pi3d_motion_target, 2, 1))

    if config.use_curriculum_loss:
        c_loss = loss_f + loss_l * pow(10, -(nb_iter - 1))
        loss = loss + c_loss
    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b, n, 36, 3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = pi3d_motion_target.reshape(b, n, 36, 3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    # add loss to wandb
    wandb.log({"loss": loss})
    # add additional information to wandb
    wandb.log({"loss_l": loss_l})
    wandb.log({"loss_f": loss_f})

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, optimizer)

    return loss.item(), optimizer, current_lr


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        job_type="train_" + args.run_name,
        mode=args.wandb_mode,
    )
    wandb.run.name = args.run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    dct_m, idct_m = get_dct_matrix(config.motion.pi3d_input_length_dct)
    dct_m = torch.tensor(dct_m).float().to(device).unsqueeze(0)  # 1,50,50
    idct_m = torch.tensor(idct_m).float().to(device).unsqueeze(0)  # 1,50,50

    # Model
    model = Model(
        input_channels=3,
        input_time_frame=config.motion.pi3d_input_length,  # 10
        output_time_frame=config.motion.pi3d_input_length,  # 10
        st_gcnn_dropout=0.1,
        joints_to_consider=36,
        n_actors=2,
        n_txcnn_layers=3,
    ).to(device)
    wandb.watch(model)

    print(
        "total number of parameters of the network is: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    wandb.log(
        {
            "number of parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        }
    )
    config.snapshot_dir = config.snapshot_dir + args.run_name
    model.train()
    model.to(device)

    # Dataset
    config.motion.pi3d_target_length = config.motion.pi3d_target_length_train
    config.data_dir = args.data
    # Training
    dataset = datasets.Datasets(opt=config, is_train=True)
    print(">>> Training dataset length: {:d}".format(dataset.__len__()))
    shuffle = True
    sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
    )
    # Validation
    eval_config = copy.deepcopy(config)
    eval_config.motion.pi3d_target_length = eval_config.motion.pi3d_target_length_eval
    actions = [
        "1/a-frame",
        "1/around-the-back",
        "1/coochie",
        "1/frog-classic",
        "1/noser",
        "1/toss-out",
        "1/cartwheel",
    ]
    eval_dataset = datasets.Datasets(config, actions, is_train=False)
    print(">>> Validation dataset length: {:d}".format(eval_dataset.__len__()))
    shuffle = False
    sampler = None
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size_test,
        num_workers=1,
        drop_last=False,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.cos_lr_max, weight_decay=config.weight_decay
    )
    ensure_dir(config.snapshot_dir)
    logger = get_logger(config.log_file, "train")
    link_file(config.log_file, config.link_log_file)

    print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

    if config.model_pth is not None:
        state_dict = torch.load(config.model_pth)
        model.load_state_dict(state_dict, strict=True)
        print_and_log_info(
            logger, "Loading model path from {} ".format(config.model_pth)
        )

    # Training
    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0

    while (nb_iter + 1) < config.cos_lr_total_iters:
        for pi3d_motion_input, pi3d_motion_target in dataloader:
            loss, optimizer, current_lr = train_step(
                pi3d_motion_input,
                pi3d_motion_target,
                model,
                optimizer,
                nb_iter,
                config.cos_lr_total_iters,
                config.cos_lr_max,
                config.cos_lr_min,
            )
            avg_loss += loss
            if (nb_iter + 1) % 100 == 0:
                avg_loss /= 100
                print(
                    "iter: {:d}, loss: {:.4f}, lr: {:.4f}".format(
                        nb_iter + 1, avg_loss, current_lr
                    )
                )
                avg_loss = 0.0

            avg_lr += current_lr

            if (nb_iter + 1) % config.save_every == 0:
                print("Saving model")
                print("Saving model to {}".format(config.snapshot_dir))

                torch.save(
                    model.state_dict(),
                    config.snapshot_dir + "/model-iter-" + str(nb_iter + 1) + ".pth",
                )
                model.eval()
                acc_tmp = test(
                    eval_config, model, eval_dataloader, device, dct_m, idct_m
                )

                # Log MPJPE to WandB
                mpjpe = acc_tmp[0]
                mpjpe_02_sec = mpjpe[0]
                mpjpe_04_sec = mpjpe[1]
                mpjpe_08_sec = mpjpe[2]
                mpjpe_1_sec = mpjpe[3]

                print(
                    "mpjpe_02_sec: {:.4f}, mpjpe_04_sec: {:.4f}, mpjpe_08_sec: {:.4f}, mpjpe_1_sec: {:.4f}".format(
                        mpjpe_02_sec, mpjpe_04_sec, mpjpe_08_sec, mpjpe_1_sec
                    )
                )

                wandb.log({"mpjpe 0.2 seconds": mpjpe_02_sec})
                wandb.log({"mpjpe 0.4 seconds": mpjpe_04_sec})
                wandb.log({"mpjpe 0.8 seconds": mpjpe_08_sec})
                wandb.log({"mpjpe 1 second": mpjpe_1_sec})

                line = ""
                for ii in acc_tmp:
                    line += str(ii) + " "
                line += "\n"
                model.train()

            if (nb_iter + 1) == config.cos_lr_total_iters:
                break
            nb_iter += 1
