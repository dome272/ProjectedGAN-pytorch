# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:43:08 2022

@author: loveboi
"""

import os
import shutil

checkpoints_path = "E:\CODING\ProjectedGAN-pytorch_FORK\checkpoints"
target_path = "E:\CODING\ProjectedGAN-pytorch_FORK\checkpoints\ckpts_outputs"
for i in range(0,77):
    ckpt_dir = os.path.join(checkpoints_path, str(i))
    imgs = [os.path.join(ckpt_dir, file) for file in os.listdir(ckpt_dir) if file.endswith(".jpg")]
    for img in imgs:
        shutil.copyfile(img, os.path.join(target_path, os.path.basename(img)))
        