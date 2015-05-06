/*
 * matrix_compute.h
 *
 *  Created: 2009-03-04, Stefan Dietze
 * Modified: 2014-05-14, René Oertel
 *
 */

#ifndef MEASURE_H_
#define MEASURE_H_

// this must be done before any stl includes due to some errors
// should be fixed in version 2.2
using namespace std;

// standard includes
#include <stdio.h>
#include <iostream>
#include <iomanip>

// include header of the CUDA Runtime API:
#include <cuda_runtime_api.h>

// \def the block size
#define BLOCK_SIZE 4

// \def the max memory for copy benchmarks
#define MAX_MEMSIZE 32*1024*1024

/** Initialize CUDA Runtime API.
 * as you should know there is no explicit init-function.
 * the API is initialized with the first API call, so this is what
 * the function should do. the function just selects the
 * first device(if there is one) and returns the properties of it.
 * \param prop the property of
 * \return the number of devices
 */
int initDevice(cudaDeviceProp *prop);

/** Print the device properties.
 * \param prop pointer to device properties structure
 * \param nr number of the device
 */
void printDeviceProperties(cudaDeviceProp *prop, int nr);

/** Measure memcopy to device.
 * \param hostPtr pointer to host memory
 * \param devPtr pointer to device memory
 * \param size size of memory
 * \return measured time
 */
float measureMemcpy2Device(float *hostPtr, float *devPtr, int size);

/** Measure memcopy from global device memory to shared memory.
 * \param devPtr pointer to global memory
 * \param size size of memory
 * \return measured time
 */
float measureGlobal2Shared(float *devPtr, int size);

/** Measure memcopy to device.
 * \param destDevPtr pointer to device memory
 * \param srcDevPtr pointer to device memory
 * \param size size of memory
 * \return measured time
 */
float measureMemcpyDevice2Device(float *destDevPtr, float *srcDevPtr,  int size);

// Benchmark function
int runBenchmark(int count);

/** benchmark-kernel
* \param devPtr pointer to memory
* \param size size of the memory
* \param time the measured time
*/
__global__ void  measureG2S_kernel(float *devPtr, int size);

#endif
