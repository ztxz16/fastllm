import numpy as np
import struct
from enum import Enum

class QuantType(Enum):
    FP32 = 0
    FP16 = 7
    INT8 = 3
    INT4 = 8

def write_int8(fo, v):
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1).clip(0.1, 1e100)
    c_scale = c_max / 127.0
    v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)
    fo.write(struct.pack('i', 3))
    fo.write(struct.pack('i', 0))
    for i in range(c_max.shape[0]):
        fo.write(struct.pack('f', -c_max[i][0]))
        fo.write(struct.pack('f', c_max[i][0]))
    fo.write(v.data)

def write_int4(fo, v):
    c_min = np.expand_dims(-np.abs(v).max(axis = -1), -1)
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
    c_scale = c_max / 7.0
    c_min = c_scale * -8.0
    v = (v - c_min) / c_scale
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)
    v = v[:, 0::2] * 16 + v[:, 1::2]
    fo.write(struct.pack('i', 8))
    fo.write(struct.pack('i', 0))
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]))
        fo.write(struct.pack('f', c_max[i][0]))
    fo.write(v.data)

class Writer():
    def __init__(self, outpath) -> None:
        self.fd = open(outpath, 'wb')
    
    def __del__(self, ):
        if not self.fd.closed:
            self.fd.close()

    def write(self, value):
        if isinstance(value, int):
            self.fd.write(struct.pack('i', value))
        elif isinstance(value, float):
            self.fd.write(struct.pack('f', value))
        elif isinstance(value, str):
            self.write_str(value)
        elif isinstance(value, bytes):
            self.write_bytes(value)
        elif isinstance(value, list):
            self.write_list(value)
        elif isinstance(value, dict):
            self.write_dict(value)
        elif isinstance(value, np.ndarray):
            self.write_tensor(value)
        else:
            raise NotImplementedError(f"Unsupport data type: {type(value)}")
    
    def write_str(self, s):
        self.write(len(s))
        self.fd.write(s.encode())

    def write_bytes(self, s):
        self.write(len(s))
        for c in s: self.write(int(c))

    def write_list(self, data):
        self.write(len(data))
        for d in data: self.write(d)
    
    def write_dict(self, data):
        self.write(len(data))
        for key in data:
            self.write_str(key)
            self.write(data[key])

    def write_tensor(self, data, data_type:QuantType=QuantType.FP32):
        self.write(list(data.shape))
        if data_type == QuantType.INT4:
            write_int4(self.fd, data)
        elif data_type == QuantType.INT8:
            write_int8(self.fd, data)
        else:
            self.write(int(data_type.value))
            self.fd.write(data.data)

