# Taken from https://github.com/NVlabs/I2SB repository under 
# the Nvidia Source Code License-NC and modified.
#
# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from diffusion_palette_eval.
#
# Source:
# https://bit.ly/eval-pix2pix
#
# ---------------------------------------------------------------

import io
import math
from PIL import Image, ImageDraw

import os

import numpy as np
import torch

from pathlib import Path
import gdown
from ipdb import set_trace as debug

FREEFORM_URL = "https://drive.google.com/file/d/1-5YRGsekjiRKQWqo0BV5RVQu0bagc12w/view?usp=share_link"

import logging
log = logging.getLogger(__name__)


# code adoptted from
# https://bit.ly/eval-pix2pix
def load_masks(filename):
    # filename = "imagenet_freeform_masks.npz"
    shape = [10000, 256, 256]

    # shape = [10950, 256, 256] # Uncomment this for places2.

    # Load the npz file.
    with open(filename, 'rb') as f:
        data = f.read()

    data = dict(np.load(io.BytesIO(data)))
    # print("Categories of masks:")
    # for key in data:
    #     print(key)

    # Unpack and reshape the masks.
    for key in data:
        data[key] = np.unpackbits(data[key], axis=None)[:np.prod(shape)].reshape(shape).astype(np.uint8)

    # data[key] contains [10000, 256, 256] array i.e. 10000 256x256 masks.
    return data

def load_freeform_masks(op_type):
    data_dir = Path("data")

    mask_fn = data_dir / f"imagenet_{op_type}_masks.npz"
    if not mask_fn.exists():
        # download orignal npz from palette google drive
        orig_mask_fn = str(data_dir / "imagenet_freeform_masks.npz")
        if not os.path.exists(orig_mask_fn):
            gdown.download(url=FREEFORM_URL, output=orig_mask_fn, quiet=False, fuzzy=True)
        masks = load_masks(orig_mask_fn)

        # store freeform of current ratio for faster loading in future
        key = {
            "freeform1020": "10-20% freeform",
            "freeform2030": "20-30% freeform",
            "freeform3040": "30-40% freeform",
        }.get(op_type)
        np.savez(mask_fn, mask=masks[key])

    # [10000, 256, 256] --> [10000, 1, 256, 256]
    return np.load(mask_fn)["mask"][:,None]


def build_inpaint_freeform(mask_type):
    assert "freeform" in mask_type

    log.info(f"[Corrupt] Inpaint: {mask_type=}  ...")

    freeform_masks = load_freeform_masks(mask_type) # [10000, 1, 256, 256]
    n_freeform_masks = freeform_masks.shape[0]
    freeform_masks = torch.from_numpy(freeform_masks)
    
    # WARN: Interpolate to 64x64
    freeform_masks = torch.nn.functional.interpolate(
        freeform_masks, size=(64, 64), mode='nearest'
    ).to(torch.float32)  # [10000, 1, 64, 64]

    def inpaint_freeform(img):
        # img: [-1,1]
        index = np.random.randint(n_freeform_masks, size=img.shape[0])
        mask = freeform_masks[index].to(img.device)
        return img * (1. - mask) + mask, mask

    return inpaint_freeform