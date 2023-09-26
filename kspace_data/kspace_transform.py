"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Modified by Nitzan Avidan, Technion - Israel Institute of Technology
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import fastmri
import numpy as np
import torch
from .subsample_kspace import MaskFunc
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pickle as pkl
import json


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()



def apply_mask(
        data: torch.Tensor,
        mask_func: MaskFunc,
        mask_type: str,
        center_fraction,
        acceleration,
        offset: Optional[int] = None,
        seed: int=0,
        padding: Optional[Sequence[int]] = None,
        idx: int = None,
        data_len: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    # print(f'idx={idx}')
    np.random.seed(idx)
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    # mask, num_low_frequencies = mask_func(shape, offset, seed)#, idx, data_len)
    num_cols = shape[-2]

    # create mask - sample the center
    num_low_frequencies = np.round(num_cols*center_fraction).astype('int32')
    mask = np.zeros(num_cols, dtype=np.float32)
    pad = ((num_cols - num_low_frequencies + 1) // 2).astype('int32')
    mask[pad: pad + num_low_frequencies] = 1
    assert mask.sum() == num_low_frequencies

    side_samples = int((num_cols / acceleration - num_low_frequencies) // 2)
    if mask_type == "fixed_equispaced" or mask_type == "offset_equispaced":
        # offset is zero
        if mask_type == "fixed_equispaced":
            offset = 0
        elif mask_type == "offset_equispaced":
            # offset depends on the
            offset = np.random.randint(0, high=round(acceleration))

        side_l_vec = np.linspace(offset, pad-1, side_samples, dtype='int')
        if side_samples*2 < (num_cols / acceleration - num_low_frequencies):
            side_samples += int((num_cols / acceleration - num_low_frequencies)) - side_samples*2
        side_r_vec = np.linspace(offset+pad + num_low_frequencies, num_cols-1, side_samples, dtype='int')
        mask[side_r_vec] = 1
        mask[side_l_vec] = 1



    elif mask_type == "random":
        samples_b = np.random.choice(pad, size=side_samples, replace=False)
        samples_e = np.random.choice(np.arange(pad+num_low_frequencies, num_cols), size=side_samples, replace=False)
        mask[samples_b] = 1
        mask[samples_e] = 1


    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1]:] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask


def apply_mask_fastmri(
        data: torch.Tensor,
        mask_func: MaskFunc,
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Optional[Sequence[int]] = None,
        idx: int = None,
        data_len: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.
    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])

    mask, num_low_frequencies = mask_func(shape, offset, seed, idx, data_len)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1]:] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.
    Applies the formula (data - mean) / (stddev + eps).
    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.
    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/
    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.
    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.
    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean,

def ifft2(kspace):
    """
    Apply ifft2 on the kspace

    Args:
        kspace: kspace tensor of shape (H,W,2)
    Returns:
        torch.Tensor: ifft on the kspace - image domain (H,W,2)
    """
    complex_kspace = torch.complex(kspace[..., 0], kspace[..., 1])
    img = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(complex_kspace, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1))
    img = torch.stack((img.real, img.imag), dim=-1)

    return img


def fft2(img):
    """
    Apply fft2 on the image domain

    Args:
        img: image tensor of shape (H,W,2)
    Returns:
        torch.Tensor: fft on the image domain - kspace (H,W,2)
    """
    complex_img = torch.complex(img[..., 0], img[..., 1])
    kspace = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(complex_img, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1))
    kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
    return kspace

def fft2_np(img):
    complex_img = (np.dstack((img[..., 0].numpy(), img[..., 1].numpy()))).view('complex128')
    kspace = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(complex_img, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1))
    kspace = np.concatenate((kspace.real, kspace.imag), axis=-1)

    return torch.from_numpy(kspace)


class KspaceUnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled kspace after inverse FFT.
        target: The target kspace (if applicable).
        mask: Mask for sampling.
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float



class KspaceUnetDataTransform:
    """
    Data Transformer for training U-Net models on the k-space.
    """

    def __init__(
            self,
            mask_type: str,
            center_fractions: float,
            accelerations: int,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
            multi2single: bool = False,
    ):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            multi2single:
        """

        self.mask_func = mask_func
        self.use_seed = use_seed
        self.mask_type = mask_type
        self.center_fractions = center_fractions
        self.accelerations = accelerations

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
            idx: int,
            data_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
            idx: Index of the image in the dataset
            multi2single: Whether to convert a multi coil data to single coil using geometric mean.
                Used only for inference.


        Returns:
            A tuple containing, undersampled k-space input, the mask, the full k-space,
            the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # inverse Fourier transform to get the image domain
        image = ifft2(kspace_torch)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        # crop the image
        image = complex_center_crop(image, crop_size)

        # multi to single:
        if self.which_challenge == 'multi2single':
            image_complex = torch.complex(image[..., 0], image[..., 1])
            gmean_coil = gmean(image_complex, axis=1)
            image = to_tensor(gmean_coil)


        kspace = fft2(image)

        # Apply log transfrom
        kspace_torch = torch.log(kspace + 1) * 1e5

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace, mask = apply_mask(kspace_torch, self.mask_func, seed=seed, idx=idx, data_len=data_len,
                                             mask_type=self.mask_type, center_fraction=self.center_fractions[0], acceleration=self.accelerations[0])
        else:
            masked_kspace = kspace_torch



 
        return KspaceUnetSample(
            image=masked_kspace.permute(2, 0, 1),
            mask=mask.permute(2, 0, 1),
            target=kspace_torch.permute(2, 0, 1),
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )

