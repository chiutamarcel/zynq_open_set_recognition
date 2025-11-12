from abc import ABC, abstractmethod
import torch
import qtensor as ql
import copy
import numpy as np
import os
import json

INPUT_DTYPE = torch.int8
WEIGHT_DTYPE = torch.int8
BIAS_DTYPE = torch.int32
ACC_DTYPE = BIAS_DTYPE # Accumulator dtype

def emit_c_float(f, name, value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Tensor {name} must contain exactly one element.")
        value = value.item()
    elif not isinstance(value, (float, np.floating)):
        raise TypeError("emit_c_float only supports float values")

    f.write(f"const float {name} = {value};\n\n")

def emit_c_array(f, name, arr, ctype):
    arr = np.array(arr)

    if ctype != "I8" and ctype != "I32" and ctype != "float":
        raise ValueError("Type must be I8, I32 or float") 
    
    # if (arr.ndim == 2):
    #     arr = arr.squeeze()

    # Choose formatting function based on type
    cast = float if ctype == "float" else int

    dim_names = [f"{name}_dim_{idx}" for idx, d in enumerate(arr.shape)]
    dims = ''.join(f'[{dn}]' for dn in dim_names)

    for idx, d in enumerate(arr.shape):
        f.write(f"const int {dim_names[idx]} = {d};\n\n")

    f.write(f"const {ctype} {name}{dims} = ")

    f.write(tensor2str(arr, cast))

    f.write(";\n\n")

def isActivationFunction(layer: torch.nn):
    if isinstance(layer, torch.nn.ReLU):
        return True
    elif isinstance(layer, torch.nn.Softmax):
        return True
    elif isinstance(layer, torch.nn.MaxPool2d):
        return True
    elif isinstance(layer, torch.nn.Flatten):
        return True
    
    return False

def tensor2str(tensor, cast):
    if tensor.ndim == 0:
        x = tensor.item()

        return str(cast(x))
    
    text = '{'

    for idx, itm in enumerate(tensor[:]):
        text += tensor2str(itm, cast)

        if idx < tensor.shape[0] - 1:
            text += ','
    
    text += '}'

    return text

class QModule():
    def __init__(self):
        self._input_min = float('inf')
        self._input_max = float('-inf')
        self.__isQuantized = False
        self.__isCalibrated = False

    def forward(self, x):
        if self.__isQuantized == False:
            raise Exception("Layer must be quantized first!")
        
    def quantize(self):
        if self.__isCalibrated == False:
            raise Exception("Layer must be calibrated first!")
        self.__isQuantized = True
        
    def calibrate(self, x):
        self.__isCalibrated = True
        self._input_min = min(self._input_min, x.min().item())
        self._input_max = max(self._input_max, x.max().item())

    def export(self, export_dir):
        if self.__isQuantized == False:
            raise Exception("Layer must be quantized first!")


class QLayer(QModule):
    def __init__(self, idx, name, layer: torch.nn.Module):
        super().__init__()
        self._idx = idx
        self._name = name
        self._layer = copy.deepcopy(layer)
        self._layer.eval()
        self._qweight = None
        self._qbias = None

    def forward(self, x):
        super().forward(x)
        

    def quantize(self):
        super().quantize()
        weights = self._layer.weight.detach()

        w_min = torch.min(weights)
        w_max = torch.max(weights)

        self._qweight = ql.quantize(self._layer.weight.detach(), min_val=w_min, max_val=w_max)
        self.input_scale = (self._input_max - self._input_min) / (2 ** 8 - 1)

        if self._layer.bias is not None:
            bias = self._layer.bias.detach()
            bias_scale = self._qweight.scale * self.input_scale
            self._qbias = ql.quantize_sub(bias, bias_scale, bit_width=32)

    def calibrate(self, x):
        super().calibrate(x)
        return self._layer(x)
    
    def export(self, export_dir):
        super().export(export_dir)
        
        filename = self._name + ".h"
        filepath = os.path.join(export_dir, filename)
        with open(filepath,'w') as f:
            f.write("#include \"common.hpp\"\n\n")
            emit_c_float(f, f"input_{self._idx}_scale", self.input_scale)
            emit_c_array(f, f"weights_{self._idx}", self._qweight.tensor.T, "I8")
            emit_c_float(f, f"weights_{self._idx}_scale", self._qweight.scale)
            emit_c_array(f, f"biases_{self._idx}", self._qbias.tensor, "I32")
            emit_c_float(f, f"biases_{self._idx}_scale", self._qbias.scale)

    def export_npy(self, export_dir):
        qweights_np = self._qweight.tensor.detach().cpu().numpy().T 
        qbias_np = self._qbias.tensor.detach().cpu().numpy()
        filename_scales = os.path.join(export_dir, "scales.json")
        
        np.save(os.path.join(export_dir, "qweights.npy"), qweights_np)
        np.save(os.path.join(export_dir, "qbias.npy"), qbias_np)

        print(f"input_scale: {self.input_scale}")
        print(f"weights_scale: {self._qweight.scale}")
        print(f"biases_scale: {self._qbias.scale}")

        scales = {
            "input_scale": self.input_scale,
            "weights_scale": float(self._qweight.scale.item()),
            "biases_scale": float(self._qbias.scale.item())
        }

        with open(filename_scales, 'w') as f:
            json.dump(scales, f)


    
class QLinearLayer(QLayer):
    def __init__(self, idx, name, layer: torch.nn.Module):
        super().__init__(idx, name, layer)

    def forward(self, x):
        super().forward(x)

        x = ql.quantize(x, min_val=self._input_min, max_val=self._input_max)
        y = torch.matmul(x.tensor.to(ACC_DTYPE), self._qweight.tensor.T.to(ACC_DTYPE)).to(ACC_DTYPE)

        if self._qbias:
            y = torch.add(y, self._qbias.tensor)
            qy = ql.qtensor_like(y, self._qbias)
        else:
            qy = ql.QTensor(y, self._qweight.scale * self.input_scale, 0, True, bit_width=32)

        return ql.dequantize(qy)
    
    def quantize(self):
        super().quantize()
    
    def calibrate(self, x):
        return super().calibrate(x)
    
    def export(self, export_dir):
        super().export(export_dir)

class QConv2D(QLayer):
    def __init__(self, idx, name, layer: torch.nn.Module):
        super().__init__(idx, name, layer)
        self.__stride = layer.stride[0]
        self.__ks = layer.kernel_size[0]
        self.__C_in = layer.in_channels
        self.__C_out = layer.out_channels
        self.__padding = layer.padding
    
    def forward(self, x):
        super().forward(x)

        if x.ndim == 3:
            x = x.unsqueeze(0)  # Add batch dimension if needed

        # qx = ql.qtensor_like(x, self._qweight)
        qx = ql.quantize(x, min_val=self._input_min, max_val=self._input_max)

        # Extract parameters
        N, C_in, H_in, W_in = x.shape
        H_out = (H_in + 2 * self.__padding[0] - self.__ks) // self.__stride + 1
        W_out = (W_in + 2 * self.__padding[1] - self.__ks) // self.__stride + 1

        # Use unfold to get sliding local blocks (im2col style)
        qx_unf = torch.nn.functional.unfold(
            qx.tensor.float(), kernel_size=self.__ks, stride=self.__stride, padding=self.__padding
        )  # shape: (N, C_in * ks * ks, L)
        qx_unf = qx_unf.to(INPUT_DTYPE)

        # Reshape weights
        qw = self._qweight.tensor.view(self.__C_out, -1)  # shape: (C_out, C_in * ks * ks)

        # Perform batched matrix multiplication
        y = torch.matmul(qw.to(ACC_DTYPE), qx_unf.to(ACC_DTYPE))  # shape: (N, C_out, L)
        y = y.view(N, self.__C_out, H_out, W_out)  # reshape to image format

        # Add bias if available
        if self._qbias:
            y += self._qbias.tensor.view(1, -1, 1, 1)

        qy = ql.qtensor_like(y, self._qbias)

        return ql.dequantize(qy)
    
    def quantize(self):
        super().quantize()
    
    def calibrate(self, x):
        return super().calibrate(x)
    
    def export(self, export_dir):
        super().export(export_dir)
        
        filename = self._name + ".h"
        filepath = os.path.join(export_dir, filename)
        with open(filepath,'w') as f:
            f.write("#include \"common.hpp\"\n\n")
            emit_c_float(f, f"input_{self._idx}_scale", self.input_scale)

            qw = np.einsum("ijkl->jikl", self._qweight.tensor)
            emit_c_array(f, f"weights_{self._idx}", qw, "I8")
            emit_c_float(f, f"weights_{self._idx}_scale", self._qweight.scale)
            emit_c_array(f, f"biases_{self._idx}", self._qbias.tensor, "I32")
            emit_c_float(f, f"biases_{self._idx}_scale", self._qbias.scale)

class QModel(QModule):
    def __init__(self, model):
        super().__init__()
        self.__model = model
        self.__children = []
        
        for idx, layer in enumerate(model):
            layer_name = f"{layer.__class__.__name__}_{idx}"
            if isinstance(layer, torch.nn.Linear):
                self.__children.append(QLinearLayer(idx, layer_name, layer))
            elif isinstance(layer, torch.nn.Conv2d):
                self.__children.append(QConv2D(idx, layer_name, layer))
            elif isActivationFunction(layer):
                self.__children.append(layer)
            else:
                raise Exception("Unsupported layer type!")
    
    def forward(self, x):
        super().forward(x)

        for child in self.__children:
            x = child.forward(x)

        return x

    def calibrate(self, x):
        super().calibrate(x)
        for child in self.__children:
            if isActivationFunction(child):
                x = child.forward(x)
            else:
                x = child.calibrate(x)

    def quantize(self):
        super().quantize()
        for child in self.__children:
            if isActivationFunction(child):
                continue

            child.quantize()

    def eval(self):
        pass # workaround to ignore model.eval()

    
    def export(self, export_dir):
        super().export(export_dir)

        for child in self.__children:
            if isActivationFunction(child):
                continue

            child.export(export_dir)

# TODO: export function
        