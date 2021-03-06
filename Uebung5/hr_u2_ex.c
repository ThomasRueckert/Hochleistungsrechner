/*
 *  Show OpenCL device info and execute kernel
 *  Copyright (C) 2010,2011 René Oertel (rene.oertel@cs.tu-chemnitz.de)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Last modifications: oere, 2011-05-24, 8:20
 *
 *  astyle --style=k/r -t8
 */

#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>		/* ENOMEM */
#include <sys/time.h>		/* gettimeofday() */

#define DEBUG 0
#define ARRAYSIZEx 1<<8
#define ARRAYSIZEy 1<<8
/*
const char oclKernelName[] = "vectorAddition";
const char *oclKernelSource[] = {
	"__kernel void vectorAddition(const __global int* a,\n"
	"                             const __global int* b,\n"
	"                                   __global int* c)\n"
	"{\n"
	"      unsigned int gid = get_global_id(0);\n"
	"      c[gid] = a[gid] + b[gid];\n"
	"}\n"
};
*/
const char oclKernelName[] = "matrixAddition";
const char *oclKernelSource[] = {
	"__kernel void matrixAddition(const __global float *a,\n"
	"                             const __global float *b,\n"
	"                                   __global float *c)\n"
	"{\n"
	"      unsigned int gid0 = get_global_id(0);\n"
	"      unsigned int gid1 = get_global_id(1);\n"
	"      unsigned int gid = gid0*get_global_size(0)+gid1;\n"
	"      c[gid] = a[gid] + b[gid];\n"
	"}\n"
};

/*
get_group_id(0)
get_local_id(1)
*/
const char oclBuildOpts[] = { "-Werror" };

cl_int showDeviceInfo(cl_device_id *);
double stopwatch(void);

int main(int argc, char *argv[])
{
	int i, j;
	char oclBuildLog[1024];
	float hostVecA[ARRAYSIZEx][ARRAYSIZEy], hostVecB[ARRAYSIZEx][ARRAYSIZEy], hostVecC[ARRAYSIZEx][ARRAYSIZEy], hostVecC2[ARRAYSIZEx][ARRAYSIZEy];

	cl_int oclErr = CL_SUCCESS;
	cl_platform_id oclPlatform;
	cl_device_id oclDevice;
	cl_device_id *oclDeviceTemp;
	size_t oclContextDevSize;
	cl_context oclContext;
	cl_command_queue oclCommandQueue;
	cl_program oclProgram;
	cl_kernel oclKernel;
	cl_context_properties oclContextProp[3] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) 1,
		0
	};
	cl_mem oclMemVecA, oclMemVecB, oclMemVecC;
	size_t oclGlobalWorkSize[2];
	oclGlobalWorkSize[0] = ARRAYSIZEx;
	oclGlobalWorkSize[1] = ARRAYSIZEy;

	/* Allocate host memory for input and output *//*
	hostVecA = (float *) malloc(sizeof(float*)*ARRAYSIZEx);
	for (i = 0; i < ARRAYSIZEx; i++) {
	  hostVecA[i] = (float) malloc(sizeof(float)*ARRAYSIZEy);
	}	
	hostVecB = (float *) malloc(sizeof(float*)*ARRAYSIZEx);
	for (i = 0; i < ARRAYSIZEx; i++) {
	  hostVecB[i] = (float) malloc(sizeof(float)*ARRAYSIZEy);
	}
	hostVecC = (float *) malloc(sizeof(float*)*ARRAYSIZEx);
	for (i = 0; i < ARRAYSIZEx; i++) {
	  hostVecC[i] = (float ) malloc(sizeof(float)*ARRAYSIZEy);
	}
	hostVecC2 = (float *) malloc(sizeof(float*)*ARRAYSIZEx);
	for (i = 0; i < ARRAYSIZEx; i++) {
	  hostVecC2[i] = (float) malloc(sizeof(float)*ARRAYSIZEy);
	}*/

	if ((NULL == hostVecA) || (NULL == hostVecB) || (NULL == hostVecC)) {
		printf
		("%s:%d - malloc of hostVecs failed - Exiting...\n",
		 __FILE__, __LINE__);
		return -ENOMEM;
	}
	
	/* Initialize memory */
	for (i = 0; i < ARRAYSIZEx; i++) {
	  for (j = 0; j < ARRAYSIZEy; j++) {
		hostVecA[i][j] = i%100;
		hostVecB[i][j] = (100-i)%100;
	  }
	}

	/* Select first platform */
	oclErr = clGetPlatformIDs(1, &oclPlatform, NULL);
	if (oclErr != CL_SUCCESS) {
		printf
		("%s:%d - clGetPlatformIDs() returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclErr);
		return -1;
	}
	oclContextProp[1] = (cl_context_properties) oclPlatform;

	/* Create context from appropriate type:
	   CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_GPU */
	oclContext =
	        clCreateContextFromType(oclContextProp, CL_DEVICE_TYPE_DEFAULT,
	                                0, 0, &oclErr);
	if (oclContext == NULL) {
		printf
		("%s:%d - clCreateContextFromType() returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Get first device information from context */
	oclErr = clGetContextInfo(oclContext, CL_CONTEXT_DEVICES, 0, NULL, &oclContextDevSize);
	if ((oclErr != CL_SUCCESS) || (oclContextDevSize == 0)) {
		printf
		("%s:%d - clGetContextInfo(CL_CONTEXT_DEVICES) returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclErr);
		return -1;
	}
	/* Work-around for multiple devices in the context */
	oclDeviceTemp = malloc(oclContextDevSize);
	if (NULL == oclDeviceTemp) {
		printf
		("%s:%d - malloc() for oclDeviceTemp failed  - Exiting...\n",
		 __FILE__, __LINE__);
		return -ENOMEM;
	}
	oclErr =
	        clGetContextInfo(oclContext, CL_CONTEXT_DEVICES,
	                         oclContextDevSize, &oclDeviceTemp[0], NULL);
	if (oclErr != CL_SUCCESS) {
		printf
		("%s:%d - clGetContextInfo(CL_CONTEXT_DEVICES) returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclErr);
		return -1;
	}
	memmove(&oclDevice, &oclDeviceTemp[0], sizeof(oclDevice));
	free(oclDeviceTemp);

#if DEBUG
	/* Show device information */
	oclErr = showDeviceInfo(&oclDevice);
	if (oclErr != CL_SUCCESS) {
		printf
		("%s:%d - showDeviceInfo returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclErr);
		return -1;
	}
#endif

	/* Create command queue */
	oclCommandQueue = clCreateCommandQueue(oclContext, oclDevice, (cl_command_queue_properties) NULL, &oclErr);
	if (oclErr != CL_SUCCESS) {
		printf
		("%s:%d - clCreateCommandQueue returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Create program object */
	oclProgram =
	        clCreateProgramWithSource(oclContext, 1, oclKernelSource, NULL,
	                                  &oclErr);
	if (CL_SUCCESS != oclErr) {
		printf
		("%s:%d - clCreateProgramWithSource() returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Build program object */
	oclErr =
	        clBuildProgram(oclProgram, 1, &oclDevice, oclBuildOpts, NULL,
	                       NULL);
	if (CL_SUCCESS != oclErr) {
		printf
		("%s:%d - clBuildProgram('%s') returned %d - Exiting...\n",
		 __FILE__, __LINE__, oclBuildOpts, oclErr);
		printf("oclKernelSource:\n-----\n%s\n-----\n",
		       oclKernelSource[0]);
		printf
		("%s:%d - clGetProgramBuildInfo() returned %d - CL_PROGRAM_BUILD_LOG:\n%s\n",
		 __FILE__, __LINE__, clGetProgramBuildInfo(oclProgram,
		                 oclDevice,
		                 CL_PROGRAM_BUILD_LOG,
		                 1024,
		                 oclBuildLog,
		                 NULL),
		 oclBuildLog);
		return -1;
	}

	/* Create kernel object */
	oclKernel = clCreateKernel(oclProgram, oclKernelName, &oclErr);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clCreateKernel() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Create input/output memory objects from host memory */
	oclMemVecA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*ARRAYSIZEx*ARRAYSIZEy, hostVecA, &oclErr);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clCreateBuffer() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}
	oclMemVecB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*ARRAYSIZEx*ARRAYSIZEy, hostVecB, &oclErr);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clCreateBuffer() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}
	oclMemVecC = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, sizeof(float)*ARRAYSIZEx*ARRAYSIZEy, NULL, &oclErr);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clCreateBuffer() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Set kernel arguments */
	oclErr = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), &oclMemVecA);
	oclErr |= clSetKernelArg(oclKernel, 1, sizeof(cl_mem), &oclMemVecB);
	oclErr |= clSetKernelArg(oclKernel, 2, sizeof(cl_mem), &oclMemVecC);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clSetKernelArg() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Execute the kernel one dimensional */
	oclErr = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, oclGlobalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clEnqueueNDRangeKernel() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Read the results back to the host memory */
	oclErr = clEnqueueReadBuffer(oclCommandQueue, oclMemVecC, CL_TRUE, 0, sizeof(float)*ARRAYSIZEx*ARRAYSIZEy, hostVecC, 0, NULL, NULL);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clEnqueueReadBuffer() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	int failure = 0;
	for (i = 0; i < ARRAYSIZEx; i++) {
	    for (j = 0; j < ARRAYSIZEy; j++) {
		hostVecC2[i][j] = hostVecA[i][j] + hostVecB[i][j];
		if (hostVecC2[i][j] != hostVecC[i][j]) {
		    printf("Error in Result at pos %d,%d\n",i,j);
		    failure++;
		}
	    }
	}
	if (failure) {
	 printf("%d Fehler\n", failure);  
	} else {
	 printf("Berechnung korrekt\n"); 
	}
	
#if DEBUG
	printf("hostVecA: ");
	for (i = 0; i < ARRAYSIZEx; i++) {
	  for (j = 0; j < ARRAYSIZEy; j++) {
	    	printf("%.3f\t", hostVecA[i][j]);
	  }
	}
	printf("\n");

	printf("hostVecB: ");
	for (i = 0; i < ARRAYSIZEx; i++) {
	  for (j = 0; j < ARRAYSIZEy; j++) {
		printf("%.3f\t", hostVecB[i][j]);
	  }
	}
	printf("\n");

	printf("hostVecC: ");
	for (i = 0; i < ARRAYSIZEx; i++) {
	  for (j = 0; j < ARRAYSIZEy; j++) {
		printf("%.3f\t", hostVecC[i][j]);
	  }
	}
	printf("\n");
	
	printf("hostVecC2: ");
	for (i = 0; i < ARRAYSIZEx; i++) {
	  for (j = 0; j < ARRAYSIZEy; j++) {
		printf("%.3f\t", hostVecC2[i][j]);
	  }
	}
	printf("\n");
#endif
	
	

	/* Clean up */
	//free(hostVecA);
	//free(hostVecB);
	//free(hostVecC);
	//printf("-- > %s:%d oclErr = %d < --\n", __FILE__, __LINE__, oclErr);
#if DEBUG
	printf("Exiting succesfully... \n");
#endif
	return 0;
}

/* Print device information */
cl_int showDeviceInfo(cl_device_id * oclDevice)
{
	cl_int oclErr = CL_SUCCESS;
	int i = 0;
	char *oclInfo = NULL;
	cl_device_type oclDeviceType = 0;

	typedef struct devinfo devinfo_t;
	struct devinfo {
		cl_device_info param_name;
		char *param_string;
	};
	devinfo_t devinfos[] = { {CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR"},
		{CL_DEVICE_NAME, "CL_DEVICE_NAME"}
	};

	oclInfo = (char *) malloc(1024 * sizeof(char));
	if (NULL == oclInfo) {
		printf("%s:%d - malloc() for clGetDeviceInfo() failed!\n",
		       __FILE__, __LINE__);
		return ENOMEM;
	}

	for (i = 0; i < 2; i++) {
		oclErr =
		        clGetDeviceInfo(*oclDevice, devinfos[i].param_name,
		                        1024, oclInfo, NULL);
		if (CL_SUCCESS != oclErr) {
			printf("%s:%d - clGetDeviceInfo() %d - failed!\n",
			       __FILE__, __LINE__, oclErr);
			free(oclInfo);
			return oclErr;
		}
		printf("%s: %s\n", devinfos[i].param_string, oclInfo);
	}
	free(oclInfo);

	oclErr =
	        clGetDeviceInfo(*oclDevice, CL_DEVICE_TYPE,
	                        sizeof(oclDeviceType), &oclDeviceType, NULL);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clGetDeviceInfo() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return oclErr;
	}

	/*
	   CL/cl.h:
	   #define CL_DEVICE_TYPE_CPU                          (1 << 1)
	   #define CL_DEVICE_TYPE_GPU                          (1 << 2)
	   #define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
	 */
	printf("CL_DEVICE_TYPE: ");
	switch ((int) oclDeviceType) {
	case CL_DEVICE_TYPE_CPU:
		printf("CL_DEVICE_TYPE_CPU\n");
		break;
	case CL_DEVICE_TYPE_GPU:
		printf("CL_DEVICE_TYPE_GPU\n");
		break;
	case CL_DEVICE_TYPE_ACCELERATOR:
		printf("CL_DEVICE_TYPE_ACCELERATOR\n");
		break;
	default:
		printf("UNKNOWN\n");
		break;
	}

	return oclErr;
}

double stopwatch(void)
{
	double diff = 0.0;

	struct timeval timenow;
	static struct timeval timeprev;	/* static to save prev. time */

	gettimeofday(&timenow, NULL);

	diff =
	        ((double) timenow.tv_sec + 1.0e-6 * (double) timenow.tv_usec) -
	        ((double) timeprev.tv_sec +
	         1.0e-6 * (double) timeprev.tv_usec);

	timeprev.tv_sec = timenow.tv_sec;
	timeprev.tv_usec = timenow.tv_usec;

	return diff;
}
