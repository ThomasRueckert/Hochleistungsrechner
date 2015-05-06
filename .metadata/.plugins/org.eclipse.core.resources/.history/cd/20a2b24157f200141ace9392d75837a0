########## the compilers and their flags #################
# the compilers
CUDACC :=  nvcc
# flags
CUDACC_FLAGS := 

########## Build Rules #################
all: measure

clean: 
	rm -f measure

measure: measure.cu
	 $(CUDACC) $(CUDACC_FLAGS) $< -o $@ $(CFLAGS)
