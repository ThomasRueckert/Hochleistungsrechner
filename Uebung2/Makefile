########## the compilers and their flags #################
# the compilers
CUDACC := nvcc
# flags
CUDACC_FLAGS :=

########## Build Rules #################
all: matrixMulFloat matrixMulDouble

clean: 
	rm -f matrixMulFloat
	rm -f matrixMulDouble

matrixMulFloat: matrixMulFloat.cu
	$(CUDACC) $(CUDACC_FLAGS) $< -o $@

matrixMulDouble: matrixMulDouble.cu
	$(CUDACC) $(CUDACC_FLAGS) $< -o $@