CC=g++
LD=ld
CXXFLAGS=-Wall -std=c++11 -g -O

NVCC=nvcc

# Space separated compute capabilities e.g. computes=70 75. If not present will automatically fetch device's compute
computes=

ifeq ($(computes), )
  computes=$(shell python get_compute.py || exit 1)
  $(info computes: $(computes))
endif

NVCCFLAGS=$(foreach compute, $(computes),-gencode arch=compute_$(compute),code=[sm_$(compute),compute_$(compute)])
$(info NVCCFLAGS: $(NVCCFLAGS))

# These are the directories where TensorRT is installed
TENSORRT_INCS=-I"/usr/include/x86_64-linux-gnu"
TENSORRT_LIBS=-L"/usr/lib/x86_64-linux-gnu"

# INCS and LIBS
INCS=-I"/usr/local/cuda/include" $(TENSORRT_INCS) -I"/usr/local/include" -I"plugin"
LIBS=-L"/usr/local/cuda/lib64" $(TENSORRT_LIBS) -L"/usr/local/lib" -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group

.PHONY: all clean

all: libyolo_layer.so

clean:
	rm -f *.so *.o

libyolo_layer.so: yolo_layer.o
	$(CC) -shared -o $@ $< $(LIBS)

yolo_layer.o: yolo_layer.cu yolo_layer.h
	$(NVCC) -ccbin $(CC) $(INCS) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ $<
