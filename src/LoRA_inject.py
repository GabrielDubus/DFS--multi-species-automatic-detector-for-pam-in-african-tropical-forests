#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 17:11:03 2025

@author: gabrieldubus
"""
from peft.tuners.lora import Linear as LoRALinear
import torch.nn as nn



def inject_lora(model, r=8, alpha=16, dropout=0.1, target_modules=["qkv", "fc1", "fc2"], adapter_name="default"):
    """
    Inject LoRA into the transformer blocks of the AST model.
    `target_modules` can be changed depending on which layers you want to apply LoRA to.
    """
    for i, block in enumerate(model.v.blocks):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                lora_module = LoRALinear(
                    base_layer=module,
                    adapter_name=adapter_name,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    fan_in_fan_out=False
                )
                # Replace the original module by LoRA-wrapped one
                parent = block
                name_parts = name.split(".")
                for part in name_parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, name_parts[-1], lora_module)

    return model