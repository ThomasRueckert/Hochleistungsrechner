########## the compilers and their flags #################
# the compilers
CUDACC := nvcc
# flags
CUDACC_FLAGS :=

########## Build Rules #################
all: matrixMulFloat

clean: 
	rm -f matrixMulFloat

matrixMulFloat: matrixMulFloat.cu
	$(CUDACC) $(CUDACC_FLAGS) $< -o $@

