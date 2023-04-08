"""
Code based on https://github.com/FraLuca/STSGCN/blob/main/model.py
"""

from typing import *
import torch
import torch.nn as nn
import math
import pdb
import numpy as np


class GCN(nn.Module):
    def __init__(self, time_dim: int, joints_dim: int, person_dim: int = 2, index: int = 0):
        """Basic module to apply graph convolution.
        https://github.com/yysijie/st-gcn/blob/master/net/utils/tgcn.py
        Args:
            time_dim (int): Time dimension i.e. number of frames.
            joints_dim (int): Joints dimension i.e. number of pose keypoints.
            person_dim (int): Person dimension i.e. number of actors. Defaults to 2.
            index (int): Id of our GCN.
        """
        super(GCN, self).__init__()
        assert person_dim > 0, "person_dim should be positive."
        self.joints_dim = joints_dim
        self.person_dim = person_dim

        # PRelu's gain
        a = 0.25
        coeffiecient = self.compute_coefficient(a)
        # ! Space matrix.
        # Create list of parameters A
        self.A = nn.Parameter(torch.FloatTensor(time_dim, joints_dim, joints_dim))

        # * In the first layer we have no activation function.
        if index == 0:
            stdv_s = math.sqrt(1/(joints_dim))
            stdv_t = math.sqrt(1/(time_dim))

        else:
            stdv_s = math.sqrt(2/(joints_dim * coeffiecient))
            stdv_t = math.sqrt(2/(time_dim * coeffiecient))
        self.A.data.uniform_(-stdv_s, stdv_s)

        # ! Temporal matrix.
        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))

        # ! We use the same of A. 
        self.T.data.uniform_(-stdv_t, stdv_t)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = torch.einsum("nctv,vtq->ncqv", (x, self.T))
        x = torch.einsum("nctv, tvw -> nctw", (x, self.A))

        return x.contiguous()

    def compute_coefficient(self, x: torch.Tensor) -> torch.Tensor:
        coeffiecient = 1+x**2

        return coeffiecient


class STGCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int],
        stride: int,
        time_dim: int,
        joints_dim: int,
        person_dim: int,
        dropout: int,
        index: int,
    ):
        """Spatio-temporal graph convolutional network.

        Args:
            in_channels (int): Size of channels in input.
            out_channels (int): Size of output channels.
            kernel_size (List[int]): Kernel size.
            stride (int): Stride.
            time_dim (int): Time dimension i.e. number of frames.
            joints_dim (int): Joints dimension i.e. number of pose keypoints.
            person_dim (int): Person dimension i.e. number of actors.
            dropout (int): Dropout probability.
        """
        super(STGCNN, self).__init__()

        self.kernel_size = kernel_size

        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1

        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        # Convolution layer.
        self.gcn = GCN(time_dim=time_dim, joints_dim=joints_dim, person_dim=person_dim, index=index)

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                stride=(stride, stride),
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x += res
        x = self.prelu(x)
        return x


class CNN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: List[int], dropout: int
    ):
        """Simple CNN layer that performs a 2D convolution maintaining the input shape, except for the features dimension.

        Args:
            in_channels (int): Size of channels in input e.g. 3 for 3D points.
            out_channels (int): Output channels.
            kernel_size (List[int]): Kernel size.
            dropout (int): Dropout probability.
        """
        super(CNN, self).__init__()

        self.kernel_size = kernel_size

        # Padding such that both dimensions are maintained.
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.s

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


# # ! CHAMPION MODEL: Classic multiple A v2
class Model(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_time_frame: int,
        output_time_frame: int,
        st_gcnn_dropout: 0.1,
        joints_to_consider: int,
        n_actors: int,
        n_txcnn_layers: int,
        txc_kernel_size: List[int],
        txc_dropout: int,
    ):
        """Spatio-temporal social graph convolutional network.

        Args:
            input_channels (int): Size of channels in input e.g. 3 for 3D points.
            input_time_frame (int): Lenght of input time frames.
            output_time_frame (int): Lenght of output time frames.
            st_gcnn_dropout (0.1): Dropout of the spatio-temporal GCNN.
            joints_to_consider (int): Number of pose keypoints in the whole scene.
            n_actors (int): Number of actors in the scene.
            n_txcnn_layers (int): Number of layers in the temporal extractor CNN.
            txc_kernel_size (List[int]): Size of the temporal extractor CNN's kernels, we have one size for all of them.
            txc_dropout (0): Dropout of the temporal extractor CNN.
        """
        super(Model, self).__init__()

        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.joints_to_consider = joints_to_consider
        self.st_gcnns = nn.ModuleList()
        self.n_txcnn_layers = n_txcnn_layers
        self.joint_per_skeleton = joints_to_consider // n_actors

        features = [input_channels, 16, 32, 64, 128, 64, 32, 16, input_channels]

        n_stcgnn_layers = len(features) - 1
        kernels = [[1, 1]] * n_stcgnn_layers
        strides = [1] * n_stcgnn_layers

        # Spatio-temporal CGNN.
        self.stgcnn = nn.Sequential()
        for n, (i, o, k, s) in enumerate(
            zip(features[:-1], features[1:], kernels, strides)
        ):
            self.stgcnn.add_module(
                name="STGCNN" + str(n),
                module=STGCNN(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=k,
                    stride=s,
                    time_dim=input_time_frame,
                    joints_dim=joints_to_consider,
                    person_dim=n_actors,
                    dropout=st_gcnn_dropout,
                    index=n,
                ),
            )

        self.fc_out = nn.Linear(self.input_time_frame, self.output_time_frame)
        nn.init.kaiming_uniform_(self.fc_out.weight, mode="fan_out")
        nn.init.constant_(self.fc_out.bias, 0)

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input pose vector.

        Returns:
            torch.Tensor: Predicted pose vector.
        """
        # B, T, V*C -> B, T, V, C.
        b, t, v = x.shape
        x = x.view(b, 3, t, v // 3)
        x = self.stgcnn(x)

        x = x.permute(0, 1, 3, 2)
        x = self.fc_out(x)
        x = x.permute(0, 3, 1, 2)
        # Reshape for compatibility.
        x = x.reshape(b, t, -1)

        

        return x

