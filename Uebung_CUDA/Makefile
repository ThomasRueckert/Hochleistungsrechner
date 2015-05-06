########## the compilers and their flags #################
# the compilers
CUDACC :=  /opt/cuda/bin/nvcc
# flags
CUDACC_FLAGS := -I/opt/cuda/include -I/usr/local/include -I/usr/include

########## Build Rules #################
all: measure

clean: 
	rm -f measure

measure: measure.cu
	 $(CUDACC) $(CUDACC_FLAGS) $< -o $@ $(CFLAGS)
