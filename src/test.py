from typing import List, Tuple
import os
from easydict import EasyDict

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import config
from datasets import pi3d as datasets
from src.model import Model
from src.utils.rigid_align import rigid_align_torch
from src.utils.visualize import vis_pi_compare
from src.utils.parser import Parser


def get_dct_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def regress_pred(
    model: nn.Module,
    pbar: DataLoader,
    num_samples: int,
    m_p3d_pi3d: np.ndarray,
    a_p3d_pi3d: np.ndarray,
    device: torch.device,
    dct_m: torch.Tensor,
    idct_m: torch.Tensor,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Regress prediction.

    Args:
        model (nn.Module): Model.
        pbar (DataLoader): Dataloader.
        num_samples (int): Number of samples.
        m_p3d_pi3d (np.ndarray): MPJPE accumulator.
        a_p3d_pi3d (np.ndarray): AME accumulator.
        device (torch.device): CUDA device.
        dct_m (torch.Tensor): DCT matrix.
        idct_m (torch.Tensor): IDCT matrix.
        visualize (bool, optional): Visualization flag. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: MPJPE and AME.
    """
    for i, (motion_input, motion_target) in enumerate(pbar):
        motion_input = motion_input.to(device)
        b, n, _ = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 36, 3)
        motion_input = motion_input[
            :,
            :,
        ].reshape(b, n, -1)
        outputs = []
        step = config.motion.pi3d_target_length_train

        # Step iterations.

        if step == config.motion.pi3d_target_length_eval:
            num_step = 1
        else:
            num_step = config.motion.pi3d_target_length_eval // step + 1  # 3

        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(
                        dct_m[:, :, : config.motion.pi3d_input_length],
                        motion_input_.to(device),
                    )
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(
                    idct_m[:, : config.motion.pi3d_input_length, :], output
                )[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:, :25]

        motion_target = motion_target[:, :25].detach()
        b, n, _ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b, 25, 36, 3)
        motion_pred = motion_target.clone().reshape(b, 25, 36, 3)
        motion_pred[
            :,
            :,
        ] = pred_rot

        motion_pred *= 1000
        motion_gt = (motion_gt.reshape(b, 25, 36, 3) * 1000).to(device)
        motion_pred = motion_pred.to(device)

        # MPJPE a.k.a. JME
        mpjpe_p3d_pi3d = torch.sum(
            torch.mean(
                torch.norm(motion_pred - motion_gt, dim=3),
                dim=2,
            ),
            dim=0,
        )
        m_p3d_pi3d += mpjpe_p3d_pi3d.cpu().numpy()

        # ! at train could give errors due to negative weights
        # * AME
        pred_ali_l = rigid_align_torch(
            motion_pred[:, :, :18, :], motion_gt[:, :, :18, :]
        ).to(device)
        pred_ali_f = rigid_align_torch(
            motion_pred[:, :, 18:, :], motion_gt[:, :, 18:, :]
        ).to(device)
        pred_ali = torch.cat((pred_ali_l, pred_ali_f), axis=2)

        ame_p3d_pi3d = torch.sum(
            torch.mean(
                torch.norm(pred_ali - motion_gt, dim=3),
                dim=2,
            ),
            dim=0,
        )
        a_p3d_pi3d += ame_p3d_pi3d.cpu().numpy()

        # * Visualization.
        path = "./viz"
        if not os.path.exists(path):
            os.makedirs(path)

        if visualize:
            p3d_gt = torch.flatten(motion_gt, start_dim=-2)
            p3d_pred = torch.flatten(motion_pred, start_dim=-2)

            for j, (gt, pred) in enumerate(zip(p3d_gt, p3d_pred)):
                vis_pi_compare(
                    p3d_gt=gt, p3d_pred=pred, save_path=path + f"/test_{j}_{i}.mp4"
                )

    m_p3d_pi3d = m_p3d_pi3d / num_samples
    a_p3d_pi3d = a_p3d_pi3d / num_samples
    return m_p3d_pi3d, a_p3d_pi3d


def test(
    config: EasyDict,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dct_m: torch.Tensor,
    idct_m: torch.Tensor,
    visualize: bool = False,
) -> Tuple[List, List]:
    """Test function.

    Args:
        config (EasyDict): Configuration object.
        model (nn.Module): Model.
        dataloader (DataLoader): DataLoader.
        device (torch.device): CPU or GPU.
        dct_m (torch.Tensor): DCT matrix.
        idct_m (torch.Tensor): IDCT matrix.
        visualize (bool, optional): Visualization flag. Defaults to False.

    Returns:
        Tuple[List, List]: MPJPE and AME.
    """
    # Metric splits
    if config.protocol in ["pro1"] + [str(x) for x in range(7)]:
        # common and single splits
        results_keys = ["#4", "#9", "#14", "#24"]
    elif config.protocol == "pro3":
        # unseen
        results_keys = ["#9", "#14", "#19"]

    # MPJPE/JME and AME
    m_p3d_pi3d = np.zeros([config.motion.pi3d_target_length_eval])
    a_p3d_pi3d = np.zeros([config.motion.pi3d_target_length_eval])

    titles = np.array(range(config.motion.pi3d_target_length_eval)) + 1
    num_samples = 0

    pbar = dataloader
    m_p3d_pi3d, a_p3d_pi3d = regress_pred(
        model,
        pbar,
        num_samples,
        m_p3d_pi3d,
        a_p3d_pi3d,
        device,
        dct_m,
        idct_m,
        visualize,
    )

    mpjpe = {}
    ame = {}
    for j in range(config.motion.pi3d_target_length_eval):
        mpjpe["#{:d}".format(titles[j])] = [m_p3d_pi3d[j], m_p3d_pi3d[j]]
        ame["#{:d}".format(titles[j])] = [a_p3d_pi3d[j], a_p3d_pi3d[j]]

    # Extract only relevant keys.
    mpjpe_out = []
    ame_out = []
    for key in results_keys:
        mpjpe_out.append(round(mpjpe[key][0], 1))
        ame_out.append(round(ame[key][1]))

    return mpjpe_out, ame_out


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    dct_m, idct_m = get_dct_matrix(config.motion.pi3d_input_length_dct)
    dct_m = torch.tensor(dct_m).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().to(device).unsqueeze(0)

    # Model
    model = Model(
        input_channels=3,
        input_time_frame=10,
        output_time_frame=10,
        st_gcnn_dropout=0.1,
        joints_to_consider=36,
        n_actors=2,
        n_txcnn_layers=3,
    )

    print(
        "total number of parameters of the network is: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    model_iter = "./" + "snapshot/model-iter-" + str(config.cos_lr_total_iters)
    config.snapshot_dir = config.snapshot_dir + model_iter
    print(config.snapshot_dir)

    state_dict = torch.load(model_iter + ".pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    # Dataset
    actions = [
        "1/a-frame",
        "1/around-the-back",
        "1/coochie",
        "1/frog-classic",
        "1/noser",
        "1/toss-out",
        "1/cartwheel",
    ]
    eval_dataset = datasets.Datasets(opt=config, is_train=False)
    print(">>> Test dataset length: {:d}".format(eval_dataset.__len__()))
    shuffle = False
    sampler = None
    dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size_test,
        num_workers=1,
        drop_last=False,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
    )

    # Inference
    mpjpe, ame = test(
        config, model, dataloader, device, dct_m, idct_m, visualize=args.visualize
    )

    print("Average MPJPE: ", mpjpe)
    print("Average AME: ", ame)
