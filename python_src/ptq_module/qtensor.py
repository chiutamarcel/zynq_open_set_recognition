#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class QTensor:
    def __init__(self, tensor: torch.Tensor, scale: float, zero_point: float, signed: bool, bit_width=8):
        tensor_type = None

        if (bit_width == 8):
            tensor_type = torch.int8
        elif (bit_width == 16):
            tensor_type = torch.int16
        elif (bit_width == 32):
            tensor_type = torch.int32
        else:
            raise ValueError("Unsupported bitwidth, available values: 8, 16, 32")

        if (signed):
            self.tensor = torch.clamp(tensor, -2**(bit_width-1), 2**(bit_width-1) - 1).to(tensor_type)
        else:
            self.tensor = torch.clamp(tensor, 0, 2**bit_width - 1).to(tensor_type)

        # self.tensor = tensor.to(torch.int32) # The quantized tensor
        self.scale = scale  # The scale used for quantization
        self.zero_point = zero_point  # The zero-point used for quantization
        self.signed = signed # Whether the quantization is signed or not
        self.bit_width = bit_width  # The bit width used for quantization


def quantize_like(x, qtensor):
    """Quantize torch Tensor x with the same quantization parameters as QTensor qtensor"""
    return quantize_sub(x, qtensor.scale, qtensor.bit_width, qtensor.signed)

def qtensor_like(qx, qtensor):
    """Create a QTensor like qtensor from qx tensor values. Observation: this function does NOT quantize qx, it assumes qx is already quantized"""
    return QTensor(qx, qtensor.scale, qtensor.zero_point, qtensor.signed, qtensor.bit_width)

def quantize(x : QTensor | torch.Tensor, signed = True, min_val=-1, max_val=1, bit_width=8):
    """Quantize x"""

    if not isinstance(x, torch.Tensor):
        raise Exception("Input must be a torch tensor or QTensor!")

    if min_val is None:
        min_val = torch.min(x)
    if max_val is None:
        max_val = torch.max(x)
    
    input_max = max_val
    input_min = min_val

    scale = (input_max - input_min) / (2 ** bit_width - 1)

    return quantize_sub(x, scale, signed=signed, bit_width=bit_width)


def quantize_sub(x, scale, bit_width=8, signed = True):
    """Quantize x with known scale"""

    if not isinstance(x, torch.Tensor):
        raise Exception("Input must be a torch tensor!")

    rez = torch.round(x / scale)

    # if (signed):
    #     rez = torch.clamp(rez, -2**(bit_width-1), 2**(bit_width-1) - 1)
    # else:
    #     rez = torch.clamp(rez, 0, 2**bit_width - 1)
    
    return QTensor(rez, scale, 0, signed, bit_width=bit_width)

def dequantize(qx: QTensor):
    """Dequantize qtensor qx"""

    if not isinstance(qx, QTensor):
        raise Exception("Input must be a QTensor!")

    return qx.scale * qx.tensor