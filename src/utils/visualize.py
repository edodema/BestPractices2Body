"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
"""
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


class ExPI3D(object):
    def __init__(self, fig: matplotlib.figure.Figure):
        self.I = np.array([0, 0, 0, 3, 4, 6, 3, 5, 7, 3, 10, 12, 14, 3, 11, 13, 15])
        self.J = np.array([1, 2, 3, 4, 6, 8, 5, 7, 9, 10, 12, 14, 16, 11, 13, 15, 17])
        self.LR = np.array(
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool
        )
        self.ax = fig.add_subplot(projection="3d")
        self.ax.cla()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim3d([-1000, 1000])  # self.x)
        self.ax.set_zlim3d([-1000, 1000])  # self.y)
        self.ax.set_ylim3d([-1000, 1000])  # self.z)
        self.ax.axis("off")

    def update(
        self,
        f: int,
        channels: torch.Tensor,
        channels2: Optional[torch.Tensor] = None,
        color_gt_l: str = "yellowgreen",
        color_gt_f: str = "orchid",
        color_pred_l: str = "darkolivegreen",
        color_pred_f: str = "darkorchid",
        plot_pred: bool = True,
    ):
        """Update the 3D skeleton.

        Args:
            f (int): Frame index.
            channels (torch.Tensor): Skeleton for gt.
            channels2 (Optional[torch.Tensor], optional): Skeleton for prediction. Defaults to None.
            color_gt_l (str, optional): Ground truth leader's color. Defaults to "yellowgreen".
            color_gt_f (str, optional): Ground truth follower's color. Defaults to "orchid".
            color_pred_l (str, optional): Prediction leader's color. Defaults to "darkolivegreen".
            color_pred_f (str, optional): Prediction follower's color. Defaults to "darkorchid".
            plot_pred (bool, optional): Plot prediction. Defaults to True.
        """
        assert (
            channels.size(dim=0) == 108
        ), "channels should have 108 for 2p, it has %d instead" % channels.size(dim=0)

        # channels: 2p for gt
        vals_ = np.reshape(channels.cpu().detach().numpy(), (36, 3))
        vals_l = vals_[:18, :]
        vals_f = vals_[18:, :]
        for j in range(2):
            vals = [vals_l, vals_f][j]
            for i in np.arange(len(self.I)):
                x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
                y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
                z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
                self.ax.plot(
                    x,
                    y,
                    z,
                    lw=2,
                    c=color_gt_l if j == 0 else color_gt_f,
                    linestyle="dashed",
                )

            # channels2: 2p for pred
            if plot_pred:
                vals2_ = np.reshape(channels2.cpu().detach().numpy(), (36, -1))
                vals2_l = vals2_[:18, :]
                vals2_f = vals2_[18:, :]
                for j in range(2):  # pred
                    vals2 = [vals2_l, vals2_f][j]
                    for i in np.arange(len(self.I)):
                        x = np.array([vals2[self.I[i], 0], vals2[self.J[i], 0]])
                        y = np.array([vals2[self.I[i], 1], vals2[self.J[i], 1]])
                        z = np.array([vals2[self.I[i], 2], vals2[self.J[i], 2]])
                        self.ax.plot(
                            x,
                            y,
                            z,
                            lw=2,
                            c=color_pred_l if j == 0 else color_pred_f,
                        )

        # time string
        if f >= 10:
            time_string = "             Pred/GT"
        else:  # input
            time_string = "             Input"
        self.ax.text2D(-0.04, -0.07, time_string, fontsize=15)


def vis_pi_compare(
    p3d_gt: torch.Tensor,
    p3d_pred: torch.Tensor,
    save_path: str,
    err_list: Optional[np.ndarray] = None,
    color_gt_l: str = "yellowgreen",
    color_gt_f: str = "orchid",
    color_pred_l: str = "darkolivegreen",
    color_pred_f: str = "darkorchid",
    plot_pred: bool = True,
):
    """Visualize the 3D pose prediction results.

    Args:
        p3d_gt (torch.Tensor): Ground truth 3D pose (50+output_n,108).
        p3d_pred (torch.Tensor): Predicted 3D pose (output_n,108).
        save_path (str): Path to save the visualization video e.g., ./outputs/test.mp4.
        err_list (np.ndarray, optional): List of errors (output_n). Defaults to None. # ?
        color_gt_l (str, optional): Ground truth leader's color. Defaults to "yellowgreen".
        color_gt_f (str, optional): Ground truth follower's color. Defaults to "orchid".
        color_pred_l (str, optional): Prediction leader's color. Defaults to "darkolivegreen".
        color_pred_f (str, optional): Prediction follower's color. Defaults to "darkorchid".
        plot_pred (bool, optional): Plot prediction. Defaults to True.
    """
    num_frames_gt = len(p3d_gt)  # 75
    num_frames_pred = len(p3d_pred)  # 25
    p3d_gt = p3d_gt.reshape((num_frames_gt, -1))
    p3d_pred = p3d_pred.reshape((num_frames_pred, -1))

    metadata = dict(title="01", artist="Matplotlib", comment="motion")
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()
    ob = ExPI3D(fig)
    with writer.saving(fig, save_path, 100):
        f = 0
        for i in tqdm(range(num_frames_gt - num_frames_pred)):  # vis input
            # * Uncomment to visualize input as well
            # ob.update(f, p3d_gt[i])
            # writer.grab_frame()
            # plt.pause(0.01)
            # plt.clf()
            f += 1
        for i in tqdm(
            range(num_frames_gt - num_frames_pred, num_frames_gt)
        ):  # vis pred vs gt
            ob.__init__(fig)
            ob.update(
                f,
                p3d_gt[i],
                p3d_pred[i - num_frames_gt + num_frames_pred],
                color_gt_l=color_gt_l,
                color_gt_f=color_gt_f,
                color_pred_l=color_pred_l,
                color_pred_f=color_pred_f,
                plot_pred=plot_pred,
            )

            # draw an error bar for err_list
            if err_list is not None:  # draw an error bar for err_list
                err = err_list[i - num_frames_gt + num_frames_pred]
                ob.ax.text2D(
                    -0.04, -0.085, "JME:" + str(round(err, 1)) + "mm", fontsize=15
                )
                fig.add_artist(
                    patches.Rectangle(
                        (0.35, 0.11),
                        0.35,
                        0.025,
                        ec="black",
                        fc="white",
                        fill=False,
                        lw=0.5,
                    )
                )
                max_err = 500
                err_len = err / max_err * 0.35
                fig.add_artist(
                    patches.Rectangle((0.35, 0.11), err_len, 0.025, fill=True, lw=0.5)
                )
            # * Uncomment to save imgs for each frame
            # fig_save_path = save_path.split(".mp4")[0]
            # plt.savefig(fig_save_path + "_" + str(i) + ".jpg")
            # plt.savefig(fig_save_path + "_" + str(i) + ".svg", dpi=350)

            writer.grab_frame()
            plt.pause(0.01)
            f += 1
    plt.close()
