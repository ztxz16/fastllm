NVCC ?= /usr/local/cuda/bin/nvcc

CUDA_HOME ?= /usr/local/cuda
TARGET := nccl_latency_bench
SRC := nccl_latency.cu

NVCCFLAGS ?= -O3 -std=c++17
LDFLAGS ?= -lnccl

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -I/usr/include -L/usr/lib/aarch64-linux-gnu $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
