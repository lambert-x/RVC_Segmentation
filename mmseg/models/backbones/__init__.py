# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

from .fan import fan_tiny_8_p4_hybrid, fan_small_12_p4_hybrid, fan_large_16_p4_hybrid
from .swin_utils import *

__all__ = [
    'fan_tiny_8_p4_hybrid', 'fan_small_12_p4_hybrid', 'fan_large_16_p4_hybrid'
]
