all: dgemm_blas dgemm_nvblas dgemm_cublas

clean: 
	rm -f dgemm_blas
	rm -f dgemm_nvblas
	rm -f nvblas.log
	rm -f dgemm_cublas

dgemm_blas: dgemm_blas.c
	gcc -Wall $< -o $@ -L../OpenBLAS/ -lopenblas -I../OpenBLAS/

run_dgemm_blas: dgemm_blas
	LD_LIBRARY_PATH=../OpenBLAS:${LD_LIBRARY_PATH} ./$<

dgemm_nvblas: dgemm_blas.c
	nvcc $< -o $@ -lnvblas -lcublas

nvblas.conf:
	@/bin/echo -e "NVBLAS_LOGFILE nvblas.log\nNVBLAS_CPU_BLAS_LIB libopenblas.so" | tee $@

run_dgemm_nvblas: dgemm_nvblas nvblas.conf
	LD_LIBRARY_PATH=../OpenBLAS:./OpenBLAS:${LD_LIBRARY_PATH} ./$<

dgemm_cublas: dgemm_cublas.c
	nvcc $< -o $@ -lcublas

run_dgemm_cublas: dgemm_cublas
	./$<
