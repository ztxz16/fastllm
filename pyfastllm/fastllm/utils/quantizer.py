import numpy as np
from enum import Enum
from .writer import Writer

class QuantType(Enum):
    FP32 = 0
    FP16 = 7
    INT8 = 3
    INT4 = 8

class Quantizer():
    quant_bit = {QuantType.FP16: 16, QuantType.INT8: 8, QuantType.INT4: 4}

    def __init__(self, quant_type:QuantType, symmetry=True) -> None:
        self.quant_type = quant_type
        self.q_bit = self.quant_bit[quant_type]

        self.up_bound = (2**(self.q_bit-1)) -1 
        self.low_bound = -(2 ** (self.q_bit-1))

        self.symmetry = symmetry

    # 范围小，单数据精度高，适用于分布集中场景
    def asymquantize(self, data:np.ndarray):
        c_min = np.expand_dims(data.min(axis=-1), -1)
        c_max = np.expand_dims(data.max(axis=-1), -1)
        c_scale = (c_max - c_min) / (self.up_bound - self.low_bound)
        c_zero = np.round(0.0 - c_min / c_scale).clip(0, self.up_bound - self.low_bound)
        c_min = -c_scale * c_zero

        q_data = (data - c_min)/ c_scale

        if self.quant_type == QuantType.FP32:
            q_data = data.astype(np.float32)
        elif self.quant_type == QuantType.FP16:
            q_data = data.astype(np.float16)
        elif self.quant_type == QuantType.INT8:
            q_data = (q_data + 0.5).astype(np.int8).clip(0, 255).astype(np.uint8)
        elif self.quant_type == QuantType.INT4:
            q_data = (q_data + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)
            q_data = q_data[:, 0::2] * 16 + q_data[:, 1::2]
        else:
            raise NotImplementedError(f"unsupport quant type")
        
        self.c_min = c_min
        self.c_max = c_max
        self.c_scale = c_scale
        self.c_zero = c_zero
        self.quant_data = q_data

        return q_data
    
    # 范围大、单数据精度低，适用分布较分散场景
    def symquantize(self, data:np.ndarray):
        c_min = np.expand_dims(-np.abs(data).max(axis = -1), -1)
        c_max = np.expand_dims(np.abs(data).max(axis = -1), -1)
        c_scale = c_max / self.up_bound
        c_min = c_scale * self.low_bound
        
        q_data = (data - c_min) / c_scale 

        if self.quant_type == QuantType.FP32:
            q_data = data.astype(np.float32)
        elif self.quant_type == QuantType.FP16:
            q_data = data.astype(np.float16)
        elif self.quant_type == QuantType.INT8:
            q_data = (q_data + 0.5).astype(np.int8).clip(1, 255).astype(np.uint8)
        elif self.quant_type == QuantType.INT4:
            q_data = (q_data + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)
            q_data = q_data[:, 0::2] * 16 + q_data[:, 1::2]
        else:
            raise NotImplementedError(f"unsupport quant type")
        
        self.c_min = c_min
        self.c_max = c_max
        self.c_scale = c_scale
        self.quant_data = q_data

        return q_data
    
    def quantize(self, data:np.ndarray):
        if self.symmetry:
            return self.symquantize(data)
        else:
            return self.asymquantize(data)
    
    def dequantize(self, ):
        if not self.c_scale: 
            raise ValueError
        
        data = self.quant_data * self.c_scale + self.c_min
        data = data.astype(np.float32)
        
        return data

    def dump(self, wt:Writer):
        wt.write(self.quant_type.value)
        if self.quant_type in (QuantType.INT4, QuantType.INT8):
            wt.write(0)
            for i in range(self.c_min.shape[0]):
                wt.write(float(self.c_min[i][0]))
                wt.write(float(self.c_max[i][0]))

        wt.fd.write(self.quant_data.data)

