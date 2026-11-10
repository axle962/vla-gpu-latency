# Build all benchmarks. Must work with nvcc from CUDA 12.4 on cuda2 (sm_75).
#   make                      # sm_75 only (CIMS cuda2)
#   make ARCHS="75 70 80 89"  # fat binary for the cluster GPUs (CUDA 12.4 has no sm_120)
NVCC   ?= nvcc
ARCHS  ?= 75
NVFLAGS ?= -O3 -std=c++17 -lineinfo
GENCODE := $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
# Also embed PTX for the newest listed arch so newer GPUs (e.g. Blackwell) can JIT it.
LAST    := $(lastword $(ARCHS))
GENCODE += -gencode arch=compute_$(LAST),code=compute_$(LAST)

BUILD  ?= build
CALIB  := device_info launch_latency bandwidth
BINS   := $(addprefix $(BUILD)/,$(CALIB))

all: $(BINS)

$(BUILD)/%: calib/%.cu common/harness.cuh | $(BUILD)
	$(NVCC) $(NVFLAGS) $(GENCODE) -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
