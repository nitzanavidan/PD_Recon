"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Modified by Nitzan Avidan, Technion - Israel Institute of Technology
"""


from fastmri.data.mri_data import SliceDataset
import numpy as np
import h5py

class KspaceDataset(SliceDataset):
    def __getitem__(self, i:int):
        fname, dataslice, metadata = self.examples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice, i, self.__len__())

        return sample


