HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_HOME :=  /opt/rocm


SOURCES = dram_bandwidth.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./demo

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)




$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) -O3 -std=c++20 -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib  -lrocprofiler64 -lrocprofiler64v2 -lhsa-runtime64 -lrocm_smi64 -ldl $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -rf /home/chenzhennan/sourceCode/YHs_Sample/amd/microbenchmark_amd/*.o
	rm -f $(HIP_PATH)/src/*.o
