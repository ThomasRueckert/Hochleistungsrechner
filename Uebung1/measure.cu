
/*
 * measure.cu
 *
 *  Created: 2009-03-04, Stefan Dietze
 * Modified: 2014-05-14, René Oertel 
 *               
 */

#include "measure.h"
//#########################################

int
main ()
{
  cudaDeviceProp props;
  if (!initDevice (&props))
    {
      std::cout << "ERROR: initDevice failed!" << std::endl;
      return -1;
    }
  printDeviceProperties (&props, 0);
  runBenchmark (10);
  return 0;
}


int
runBenchmark (int count)
{

  float erg;
  int size = MAX_MEMSIZE;
  float *data = new float[size];
  int sizeB = sizeof (float) * size;
  cudaError_t ret;
  float *dPtr;
  ret = cudaMalloc ((void **) &dPtr, sizeB);
  if (ret != cudaSuccess)
    {
      std::cout << "ERROR: Malloc of device memory failed!" << std::endl;
      return -1;
    }
  else
    std::cout << sizeB << " Bytes of global memory allocated" << std::endl;

  //---------------------------Test 1-------------------------------------
  std::cout << "==========================================\n";
  std::cout << "Benchmark: Memcpy Host to Device" << std::endl;
  for (int stepSize = (1024 * 1024); stepSize <= (sizeB); stepSize *= 2)
    {

      erg = 0;
      for (int i = 0; i < count; i++)
	erg += measureMemcpy2Device (data, dPtr, stepSize);
      erg = erg / count;
      double speed = 0;
      float sizeMB = stepSize / (1024 * 1024);
      if (erg != 0)
	speed = (sizeMB / erg) * 1000;
      std::cout << "Result: copied " << sizeMB << " MB in " << erg <<
	" ms with " << speed << " MB/s" << std::endl;
    }
  // free host memory
  delete data;
  std::cout << "==========================================" << std::endl <<
    std::endl;
  //---------------------------Test 2-------------------------------------
  ret = cudaMallocHost ((void **) &data, sizeB);
  if (ret != cudaSuccess)
    {
      std::cout << "ERROR: Malloc of page-locked memory failed!" << std::endl;
      return -1;
    }
  else
    std::cout << sizeB << " Bytes of page-locked host memory allocated"
      << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Benchmark: Memcpy Host to Device" << std::endl <<
    "(page-locked host memory)" << std::endl;
  for (int stepSize = (1024 * 1024); stepSize <= (sizeB); stepSize *= 2)
    {

      erg = 0;
      for (int i = 0; i < count; i++)
	erg += measureMemcpy2Device (data, dPtr, stepSize);
      erg = erg / count;
      double speed = 0;
      float sizeMB = stepSize / (1024 * 1024);
      if (erg != 0)
	speed = (sizeMB / erg) * 1000;
      std::cout << "Result: copied " << sizeMB << " MB in " << erg <<
	" ms with " << speed << " MB/s" << std::endl;
    }
  std::cout << "==========================================" << std::endl <<
    std::endl;
  //---------------------------Test 3-------------------------------------
  float *dPtr2;
  ret = cudaMalloc ((void **) &dPtr2, sizeB);
  if (ret != cudaSuccess)
    {
      std::cout << "ERROR: Malloc of device memory failed!" << std::endl;
      return -1;
    }
  else
    std::cout << sizeB << " Bytes of global memory allocated" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Benchmark: Memcpy Device to Device" << std::endl <<
    "(global to global)" << std::endl;
  for (int stepSize = (1024 * 1024); stepSize <= (sizeB); stepSize *= 2)
    {

      erg = 0;
      for (int i = 0; i < count; i++)
	erg += measureMemcpyDevice2Device (dPtr2, dPtr, stepSize);
      erg = erg / count;
      double speed = 0;
      float sizeMB = stepSize / (1024 * 1024);
      if (erg != 0)
	speed = (sizeMB / erg) * 1000;
      std::cout << "Result: copied " << sizeMB << " MB in " << erg <<
	" ms with " << speed << " MB/s" << std::endl;
    }
  std::cout << "==========================================" << std::endl <<
    std::endl;
  //---------------------------Test 4-------------------------------------
  size = 2048;
  sizeB = size * sizeof (float);
  std::cout << "==========================================" << std::endl;
  std::cout <<
    "Benchmark: Memcpy Device to Device \n(global to shared memory) \n copying "
    << sizeB << " Bytes ...." << std::endl;
  for (int i = 0; i < count; i++)
    erg += measureGlobal2Shared (dPtr, size);

  erg = erg / count;
  double speed = 0;
  float sizeKB = sizeB / (1024);
  if (erg != 0)
    speed = (sizeKB / erg) * 1000;
  std::cout << "Result: copied in " << erg << "ms  "
    << " with " << speed << "KB/s" << std::endl;

  cudaFree (dPtr);
  return 1;
}

//#########################################
float
measureMemcpyDevice2Device (float *destDevPtr, float *srcDevPtr, int size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaError_t error = cudaMemcpy( destDevPtr, srcDevPtr, size, cudaMemcpyDeviceToDevice);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
  // TODO: copy size bytes from destDevPtr to srcDevPtr in global memory and return the time measured
}

//#########################################
float
measureMemcpy2Device (float *hostPtr, float *devPtr, int size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaError_t error = cudaMemcpy( devPtr, hostPtr, size, cudaMemcpyHostToDevice);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
  // TODO: copy size bytes from hostPtr to devPtr and return the time measured
}

float
measureGlobal2Shared (float *devPtr, int size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Festlegung der Größe der Thread-Blocks und des Grids:
	dim3 dimBlock(8, 8);
	dim3 dimGrid(2, 2);
	// Aufruf des Kernels (Execution Configuration):
	measureG2S_kernel<<< dimGrid, dimBlock >>>(devPtr, size);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
  // TODO: start the kernel measureG2S_kernel and return the measured execution time
}

//#######################################
/*
 * Name der Grafikkarte
 * Compute Capability
 * Speichergröße
 * Anzahl Multiprozessoren
 * Register pro Block
 * Warpsize
 */
int
initDevice (cudaDeviceProp * prop)
{
	int count;
	cudaError_t error =  cudaGetDeviceCount(&count);
	if (error == cudaSuccess) {
		if (count > 0) {
			error = cudaSetDevice(0);
			if (error == cudaSuccess) {
				//device found -> write the properties inside prop
				error = cudaGetDeviceProperties(prop, 0);
			}
		}
	}
	return count;
}

//#########################################
void
printDeviceProperties (cudaDeviceProp * prop, int nr)
{
  using namespace std;
  cout << "------------Device Properties-------------" << endl;
  cout << "------------------------------------------" << endl;
  cout << left << setw (25) << "Device:" << nr << endl;
  cout << left << setw (25) << "Name:" << prop->name << endl;
  cout << left << setw (25) << "Compute Capability:" << prop->major << "." <<
    prop->minor << endl;
  cout << left << setw (25) << "Global Memory:" << prop->totalGlobalMem /
    (1024 * 1024) << " MB" << endl;
  cout << left << setw (25) << "Clock rate:" << prop->clockRate /
    1000 << " MHz" << endl;
  cout << left << setw (25) << "Multiprocessor count:" <<
    prop->multiProcessorCount << endl;
  cout << left << setw (25) << "Max. threads per block:" <<
    prop->maxThreadsPerBlock << endl;
  cout << left << setw (25) << "Max. threads per dim:" <<
    prop->
    maxThreadsDim[0] << " x " << prop->maxThreadsDim[1] << " x " << prop->
    maxThreadsDim[2] << endl;
  cout << left << setw (25) << "Max. grid size:" << prop->
    maxGridSize[0] << " x " << prop->maxGridSize[1] << " x " << prop->
    maxGridSize[2] << endl;
  cout << left << setw (25) << "Shared Memory per block:" << prop->
    sharedMemPerBlock << " Byte" << endl;
  cout << left << setw (25) << "Register per block:" << prop->
    regsPerBlock << endl;
  cout << left << setw (25) << "Warp size:" << prop->warpSize << endl;
  cout << left << setw (25) << "Memory pitch:" << prop->memPitch << endl;
  cout << "------------------------------------------" << endl << endl;
}

//#########################################
// used to get offset address of shared memory
extern __shared__ char startOfShMem[];
__global__ void
measureG2S_kernel (float *devPtr, int size)
{
  int n = blockDim.x;
  const int s = size / n;
  // define memory 
  float *shared = (float *) &startOfShMem;
  // make sure everyone is ready
  __syncthreads ();
  // now everyone copies one piece of a "line" and syncs
  for (int i = 0; i < s; i++)
    {
      shared[i] = devPtr[i * n + threadIdx.x];
      __syncthreads ();
    }
}
