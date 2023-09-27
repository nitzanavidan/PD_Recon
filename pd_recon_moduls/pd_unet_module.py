"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Modified by Nitzan Avidan, Technion - Israel Institute of Technology
"""


from argparse import ArgumentParser
import fastmri
import torch
from torch.nn import functional as F

from .mri_module import MriModule


class ResUnetKspaceModule(MriModule):
    """
    training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
            self,
            model_type,
            in_chans=2,
            out_chans=2,
            chans=32,
            num_pool_layers=4,
            drop_prob=0.0,
            lr=0.001,
            lr_step_size=40,
            lr_gamma=0.1,
            reg=0.9,
            weight_decay=0.0,
            **kwargs,
    ):
        """
        Args:
            model_type: The model used for train/val/test.
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.reg = reg
        self.model = model_type(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

    def forward(self, kspace, mask):
        return self.model(kspace, mask)

    def training_step(self, batch, batch_idx):
        output = self(batch.image, batch.mask)

        loss = F.l1_loss(output, batch.target)
        self.log("loss", loss.detach())

        return loss

        

    def validation_step(self, batch, batch_idx):
        print('validation step')
        output = self(batch.image, batch.mask)

        loss = F.l1_loss(output, batch.target) 


        # relog
        output = torch.exp(output / 1e5) - 1
        inputs = (torch.exp(batch.image / 1e5) - 1) * batch.mask
        target = torch.exp(batch.target / 1e5) - 1

        # convert 2 channels kspace (bs,2,320,320) to image domain (1,320,320)
        output = fastmri.complex_abs(fastmri.ifft2c(output.permute(0, 2, 3, 1)))
        inputs = fastmri.complex_abs(fastmri.ifft2c(inputs.permute(0, 2, 3, 1)))
        target = fastmri.complex_abs(fastmri.ifft2c(target.permute(0, 2, 3, 1)))

        kspace_data = abs(output[ 0, ...])

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "input": inputs,  # added
            "kspace_data": kspace_data,  # added
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        print('test')
        output = self.forward(batch.image)

        # convert 2 channels kspace (bs,2,320,320) to image domain (1,320,320)
        output = fastmri.complex_abs(fastmri.ifft2c(output.permute(0, 2, 3, 1)))

        output = self.forward(batch.kspace, batch.mask)
        kspace_data = abs(output[:, 0, ...])

        # relog
        output = torch.exp(output / 1e5) - 1

        # convert 2 channels kspace (bs,2,320,320) to image domain (1,320,320)
        output = fastmri.complex_abs(fastmri.ifft2c(output.permute(0, 2, 3, 1)))


        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=2, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
