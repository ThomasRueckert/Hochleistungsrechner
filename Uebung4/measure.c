/*
 *  Show OpenCL device info and measure memory bandwidth
 *  Copyright (C) 2010 René Oertel (rene.oertel@cs.tu-chemnitz.de)
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
 *  Last modifications: oere, 2011-05-10, 16:23
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>		/* atoi() */
#include <errno.h>		/* ENOMEM */
#include <getopt.h>
#include <sys/time.h>		/* gettimeofday() */

#define MAX_MEM_SIZE_H2D (1 << 28)	/* 256 MiBytes */
#define MAX_MEM_SIZE_D2H (1 << 28)
#define MAX_MEM_SIZE_D2D (1 << 27)	/* 128 MiBytes */

#define ITERATIONS 10		/* Number of measurement repetitions */

void usage(int, char **);
void parse_opts(int, char **);
double stopwatch(void);

cl_int initPlatform(cl_platform_id *, unsigned int);
cl_int initDevice(cl_platform_id *, cl_device_id *, unsigned int);
cl_int initContext(cl_device_id *, cl_context *);
cl_int showDeviceInfo(cl_device_id *);
cl_int measureBandwidth(cl_device_id *, cl_context *);
cl_int measureBandwidthH2D(cl_context *, cl_command_queue *);
cl_int measureBandwidthD2H(cl_context *, cl_command_queue *);
cl_int measureBandwidthD2D(cl_context *, cl_command_queue *);
cl_int measureBandwidthH2Dsized(cl_context *, cl_command_queue *, size_t);
cl_int measureBandwidthD2Hsized(cl_context *, cl_command_queue *, size_t);
cl_int measureBandwidthD2Dsized(cl_context *, cl_command_queue *, size_t);

typedef struct devinfo devinfo_t;
struct devinfo {
	cl_device_info param_name;
	char *param_string;
};

/* Default values for platform and device IDs */
int plat = 0;
int dev = 0;

int main(int argc, char *argv[])
{
	cl_platform_id platid;
	cl_device_id devid;
	cl_context devctx;
	cl_int err = CL_SUCCESS;

	parse_opts(argc, argv);

	err = initPlatform(&platid, plat);
	if (CL_SUCCESS != err) {
		printf("%s:%d - initPlatform() failed!\n", __FILE__,
		       __LINE__);
		return err;
	}

	err = initDevice(&platid, &devid, dev);
	if (CL_SUCCESS != err) {
		printf("%s:%d - initDevice() failed!\n", __FILE__,
		       __LINE__);
		return err;
	}

	err = initContext(&devid, &devctx);
	if (CL_SUCCESS != err) {
		printf("%s:%d - initContext() failed!\n", __FILE__,
		       __LINE__);
		return err;
	}

	err = showDeviceInfo(&devid);
	if (CL_SUCCESS != err) {
		printf("%s:%d - showDeviceInfo() failed!\n", __FILE__,
		       __LINE__);
		return err;
	}

	err = measureBandwidth(&devid, &devctx);
	if (CL_SUCCESS != err) {
		printf("%s:%d - measureBandwidth() failed!\n", __FILE__,
		       __LINE__);
		return err;
	}

	return err;
}

/* Print usage information */
void usage(int argc, char *argv[])
{

	printf("\nUsage: %s [OPTION]...\n"
	       "Available arguments:\n\n"
	       "\t-p/--platform=<ID> Choose platform (Default: 0)\n"
	       "\t-d/--device=<ID> Choose device (Default: 0)\n", argv[0]
	    );

	exit(0);
}

/* Parse commandline arguments */
void parse_opts(int argc, char *argv[])
{
	int option = 0;
	const char *short_options = "p:d:h";
	const struct option long_options[] = {
		{"platform", 1, NULL, 'p'},
		{"device", 1, NULL, 'd'},
		{"help", 0, NULL, 'h'},
		{0, 0, 0, 0}
	};

	do {
		option = getopt_long(argc, argv, short_options,
				     long_options, NULL);

		switch (option) {
		case 'p':
			plat = atoi(optarg);
			printf("Platform %d selected\n", plat);
			break;

		case 'd':
			dev = atoi(optarg);
			printf("Device %d selected\n", dev);
			break;
		case 'h':	/* -h or --help */
			usage(argc, argv);
			break;
		default:
			break;
		}

	}
	while (-1 != option);

}

/* Initialize platform */
cl_int initPlatform(cl_platform_id * platid, unsigned int platnr)
{
	cl_int err = CL_SUCCESS;
	cl_uint num_platforms = 0;
	cl_platform_id *platforms = NULL;

	// Determine number of OpenCL platforms
	err = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		return err;
	}

	if (0 == num_platforms) {
		printf("%s:%d - No OpenCL platform found!\n", __FILE__,
		       __LINE__);
		return CL_INVALID_PLATFORM;
	}

	if (0 == platnr) {	/* We will use the first/default platform */
		err = /* FIXME */ ;
		if (CL_SUCCESS != err) {
			printf("%s:%d - Default FIXME() failed!\n",
			       __FILE__, __LINE__);
			return err;
		}

		return err;

	} else {		/* Not using the default platform */
		if (platnr > num_platforms - 1) {
			printf
			    ("%s:%d - Selected platform (%d) not found!\n",
			     __FILE__, __LINE__, platnr);
			return CL_INVALID_PLATFORM;
		}

		platforms =
		    (cl_platform_id *) malloc(num_platforms *
					      sizeof(cl_platform_id));
		if (NULL == platforms) {
			printf("%s:%d - malloc() for FIXME() failed!\n",
			       __FILE__, __LINE__);
			return ENOMEM;
		}

		err = /* FIXME */ ;
		if (CL_SUCCESS != err) {
			printf("%s:%d - Selected FIXME() failed!\n",
			       __FILE__, __LINE__);
			free(platforms);
			return err;
		}

		/* Save platform ID */
		*platid = platforms[platnr];
		free(platforms);

		return err;
	}
}

cl_int initDevice(cl_platform_id * platid, cl_device_id * devid,
		  unsigned int devnr)
{
	cl_int err = CL_SUCCESS;
	cl_uint num_devices = 0;
	cl_device_id *devices = NULL;

	// Determine number of OpenCL devices in this platform
	err = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		return err;
	}

	if (0 == num_devices) {
		printf("%s:%d - No OpenCL device found!\n", __FILE__,
		       __LINE__);
		return CL_INVALID_DEVICE;
	}

	if (0 == devnr) {	/* We will use the first/default device */
		err = /* FIXME */ ;
		if (CL_SUCCESS != err) {
			printf("%s:%d - Default FIXME() failed!\n",
			       __FILE__, __LINE__);
			return err;
		}

		return err;

	} else {		/* Not using the default device */
		if (devnr > num_devices - 1) {
			printf("%s:%d - Selected device (%d) not found!\n",
			       __FILE__, __LINE__, devnr);
			return CL_INVALID_DEVICE;
		}

		devices =
		    (cl_device_id *) malloc(num_devices *
					    sizeof(cl_device_id));
		if (NULL == devices) {
			printf("%s:%d - malloc() for FIXME() failed!\n",
			       __FILE__, __LINE__);
			return ENOMEM;
		}

		err = /* FIXME */ ;
		if (CL_SUCCESS != err) {
			printf("%s:%d - Selected FIXME() failed!\n",
			       __FILE__, __LINE__);
			free(devices);
			return err;
		}

		/* Save device ID */
		*devid = devices[devnr];
		free(devices);

		return err;
	}
}

/* Initialize context for device */
cl_int initContext(cl_device_id * devid, cl_context * devctx)
{
	cl_int err = CL_SUCCESS;

	/* Create context for device */
	*devctx = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		return err;
	}

	return err;
}

/* Print device information */
cl_int showDeviceInfo(cl_device_id * devid)
{
	cl_int err = CL_SUCCESS;
	cl_ulong memsize = 0;
	int i = 0;
	char *infostr = NULL;
	/* Determine vendor, device name, device version, driver version and
	   device extensions */   
	devinfo_t devinfos[] = { {CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR"},
	{ CL_DEVICE_NAME , "CL_DEVICE_NAME"},
	{ CL_DEVICE_VERSION , "CL_DEVICE_VERSION"},
	{ CL_DRIVER_VERSION , "CL_DRIVER_VERSION"},
	{ CL_DEVICE_EXTENSIONS , "CL_DEVICE_EXTENSIONS"}
	};

	/* Allocate buffer for information strings */
	infostr = (char *) malloc(1024 * sizeof(char));
	if (NULL == infostr) {
		printf("%s:%d - malloc() for infostr failed!\n",
		       __FILE__, __LINE__);
		return ENOMEM;
	}

	for (i = 0; i < 5; i++) {
		err = clGetDeviceInfo(devid, devinfos[i].param_name, sizeof(infostr), infostr, NULL);
		    if (CL_SUCCESS != err) {
			printf("%s:%d - clGetDeviceInfo() (%s) failed!\n",
			       __FILE__, __LINE__, devinfos[i].param_string);
			free(infostr);
			return err;
		}
		printf("%s: %s\n", devinfos[i].param_string, infostr);
	}
	free(infostr);

	/* Determine local memory size */
	err = clGetDeviceInfo(devid, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(memsize), memsize, NULL);
	if (CL_SUCCESS != err) {
		printf("%s:%d - clGetDeviceInfo() (CL_DEVICE_LOCAL_MEM_SIZE)  failed!\n", __FILE__, __LINE__);
		return err;
	}
	printf("CL_DEVICE_LOCAL_MEM_SIZE: %lu Bytes\n", memsize);

	/* Reset memory size variable */
	memsize = 0;
  
	/* Determine global memory size */
	err = clGetDeviceInfo(devid, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memsize), memsize, NULL);
	if (CL_SUCCESS != err) {
		printf("%s:%d - clGetDeviceInfo() (CL_DEVICE_GLOBAL_MEM_SIZE) failed!\n", __FILE__, __LINE__);
		return err;
	}
	printf("CL_DEVICE_GLOBAL_MEM_SIZE: %lu Bytes\n", memsize);

	return err;
}

/* Main function for measurements */
cl_int measureBandwidth(cl_device_id * devid, cl_context * devctx)
{
	cl_int err = CL_SUCCESS;
	cl_command_queue queue;

	/* Create command queue for device and context */
	queue = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		return err;
	}

	/* Host to device measurement */
	printf("Host to device:\n");
	err = measureBandwidthH2D(devctx, &queue);
	if (CL_SUCCESS != err) {
		printf("%s:%d - measureBandwidthH2D() failed!\n",
		       __FILE__, __LINE__);
		clReleaseCommandQueue(queue);
		return err;
	}

	/* Device to host measurement */
	printf("Device to host:\n");
	err = measureBandwidthD2H(devctx, &queue);
	if (CL_SUCCESS != err) {
		printf("%s:%d - measureBandwidthD2H() failed!\n",
		       __FILE__, __LINE__);
		clReleaseCommandQueue(queue);
		return err;
	}

	/* Device to device measurement */
	printf("Device to device:\n");
	err = measureBandwidthD2D(devctx, &queue);
	if (CL_SUCCESS != err) {
		printf("%s:%d - measureBandwidthD2D() failed!\n",
		       __FILE__, __LINE__);
		clReleaseCommandQueue(queue);
		return err;
	}

	clReleaseCommandQueue(queue);
	return err;
}

/* Loops over several buffer sizes for host to device transfers */
cl_int measureBandwidthH2D(cl_context * devctx, cl_command_queue * devq)
{
	cl_int err = CL_SUCCESS;
	size_t sz = 0;

	for (sz = 1; sz <= MAX_MEM_SIZE_H2D; sz = sz << 1) {
		if (CL_SUCCESS != FIXME(devctx, devq, sz))
			break;
	}

	return err;
}

/* Loops over several buffer sizes for device to host transfers */
cl_int measureBandwidthD2H(cl_context * devctx, cl_command_queue * devq)
{
	cl_int err = CL_SUCCESS;
	size_t sz = 0;

	for (sz = 1; sz <= MAX_MEM_SIZE_D2H; sz = sz << 1) {
		if (CL_SUCCESS != FIXME(devctx, devq, sz))
			break;
	}

	return err;
}

/* Loops over several buffer sizes for device to device transfers */
cl_int measureBandwidthD2D(cl_context * devctx, cl_command_queue * devq)
{
	cl_int err = CL_SUCCESS;
	size_t sz = 0;

	for (sz = 1; sz <= MAX_MEM_SIZE_D2D; sz = sz << 1) {
		if (CL_SUCCESS != FIXME(devctx, devq, sz))
			break;
	}

	return err;
}

/* Host to device measurement */
cl_int measureBandwidthH2Dsized(cl_context * devctx,
				cl_command_queue * devq, size_t size)
{
	cl_int err = CL_SUCCESS;
	cl_mem ddata = NULL;
	unsigned char *hdata = NULL;
	size_t sz = 0;
	unsigned int i = 0;
	double time = 0.0;
	double bandwidth = 0.0;

	/* Allocate host buffer */
	hdata = (unsigned char *) malloc(size * sizeof(unsigned char));
	if (NULL == hdata) {
		printf("%s:%d - malloc() for measureBandwidthH2D()"
		       " of %u Bytes failed!\n", __FILE__, __LINE__,
		       (unsigned int) size);
		return ENOMEM;
	}

	/* Initialize host buffer */
	for (sz = 0; sz < size; sz++) {
		hdata[i] = (unsigned char) (sz & 0xff);
	}

	/* Create device buffer */
	ddata = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		free(hdata);
		return err;
	}

	/* Synchronize */
	err = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - clFinish() failed!\n", __FILE__, __LINE__);
		free(hdata);
		clReleaseMemObject(ddata);
		return err;
	}

	/* Start the stopwatch */
	time = stopwatch();

	/* Copy data from host to device */
	for (i = 0; i < ITERATIONS; i++) {
		err =		/* FIXME */
		    if (CL_SUCCESS != err) {
			printf("%s:%d - clEnqueueFIXMEBuffer() failed!\n",
			       __FILE__, __LINE__);
			clReleaseMemObject(ddata);
			free(hdata);
			return err;
		}
	}

	/* Wait for queue finish */
	err = clFinish(*devq);

	/* Stop the stopwatch :) */
	time = stopwatch();
	if (CL_SUCCESS != err) {
		printf("%s:%d - clFinish() failed!\n", __FILE__, __LINE__);
		free(hdata);
		clReleaseMemObject(ddata);
		return err;
	}

	/* Calculate the memory bandwidth */
	bandwidth =
	    ((double) size * (double) ITERATIONS) / (time *
						     (double) (1 << 20));

	printf("%u %f\n", (unsigned int) size, bandwidth);

	clReleaseMemObject(ddata);
	free(hdata);

	return err;
}

/* Device to host measurement */
cl_int measureBandwidthD2Hsized(cl_context * devctx,
				cl_command_queue * devq, size_t size)
{
	cl_int err = CL_SUCCESS;
	cl_mem ddata = NULL;
	unsigned char *hdata = NULL;
	size_t sz = 0;
	unsigned int i = 0;
	double time = 0.0;
	double bandwidth = 0.0;

	/* Allocate host memory */
	hdata = (unsigned char *) malloc(size * sizeof(unsigned char));
	if (NULL == hdata) {
		printf("%s:%d - malloc() for measureBandwidthD2H()"
		       " of %u Bytes failed!\n", __FILE__, __LINE__,
		       (unsigned int) size);
		return ENOMEM;
	}

	/* Initialize host memory */
	for (sz = 0; sz < size; sz++) {
		hdata[i] = (unsigned char) (sz & 0xff);
	}

	/* Create device buffer */
	ddata = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		free(hdata);
		return err;
	}

	/* Initialize device memory with host buffer */
	err = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - clEnqueueFIXMEBuffer() failed!\n",
		       __FILE__, __LINE__);
		clReleaseMemObject(ddata);
		free(hdata);
		return err;
	}

	/* Synchronize */
	err = clFinish(*devq);
	if (CL_SUCCESS != err) {
		printf("%s:%d - clFinish() failed!\n", __FILE__, __LINE__);
		clReleaseMemObject(ddata);
		free(hdata);
		return err;
	}

	/* Start the stopwatch */
	time = stopwatch();

	/* Copy buffer from device to host memory */
	for (i = 0; i < ITERATIONS; i++) {
		err = /* FIXME */ ;
		if (CL_SUCCESS != err) {
			printf("%s:%d - clEnqueueFIXMEBuffer() failed!\n",
			       __FILE__, __LINE__);
			clReleaseMemObject(ddata);
			free(hdata);
			return err;
		}
	}

	err = clFinish(*devq);
	/* Synchronize and stop the stopwatch */
	time = stopwatch();
	if (CL_SUCCESS != err) {
		printf("%s:%d - clFinish() failed!\n", __FILE__, __LINE__);
		clReleaseMemObject(ddata);
		free(hdata);
		return err;
	}

	/* Calculate memory bandwidth */
	bandwidth =
	    ((double) size * (double) ITERATIONS) / (time *
						     (double) (1 << 20));

	printf("%u %f\n", (unsigned int) size, bandwidth);

	clReleaseMemObject(ddata);
	free(hdata);

	return err;
}

/* Device to device measurement */
cl_int measureBandwidthD2Dsized(cl_context * devctx,
				cl_command_queue * devq, size_t size)
{
	cl_int err = CL_SUCCESS;
	cl_mem dsource = NULL;
	cl_mem ddest = NULL;
	unsigned char *hdata = NULL;
	size_t sz = 0;
	unsigned int i = 0;
	double time = 0.0;
	double bandwidth = 0.0;

	/* Allocate host memory */
	hdata = (unsigned char *) malloc(size * sizeof(unsigned char));
	if (NULL == hdata) {
		printf("%s:%d - malloc() for measureBandwidthD2D()"
		       " of %u Bytes failed!\n", __FILE__, __LINE__,
		       (unsigned int) size);
		return ENOMEM;
	}

	/* Initialize host memory */
	for (sz = 0; sz < size; sz++) {
		hdata[i] = (unsigned char) (sz & 0xff);
	}

	/* Create device source buffer */
	dsource = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		free(hdata);
		return err;
	}

	/* Create device destination buffer */
	ddest = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		clReleaseMemObject(dsource);
		free(hdata);
		return err;
	}

	/* Initialize device source buffer with host buffer data */
	err = /* FIXME */ ;
	if (CL_SUCCESS != err) {
		printf("%s:%d - FIXME() failed!\n", __FILE__, __LINE__);
		clReleaseMemObject(ddest);
		clReleaseMemObject(dsource);
		free(hdata);
		return err;
	}

	/* Synchronize */
	err = clFinish(*devq);
	if (CL_SUCCESS != err) {
		printf("%s:%d - clFinish() failed!\n", __FILE__, __LINE__);
		clReleaseMemObject(ddest);
		clReleaseMemObject(dsource);
		free(hdata);
		return err;
	}

	/* Start the stopwatch */
	time = stopwatch();

	/* Copy from source to destination device buffer */
	for (i = 0; i < ITERATIONS; i++) {
		err =		/* FIXME */
		    if (CL_SUCCESS != err) {
			printf("%s:%d - clEnqueueFIXMEBuffer() failed!\n",
			       __FILE__, __LINE__);
			clReleaseMemObject(ddest);
			clReleaseMemObject(dsource);
			free(hdata);
			return err;
		}
	}

	err = clFinish(*devq);
	/* Synchronize and stop the stopwatch */
	time = stopwatch();
	if (CL_SUCCESS != err) {
		printf("%s:%d - clFinish() failed!\n", __FILE__, __LINE__);
		clReleaseMemObject(ddest);
		clReleaseMemObject(dsource);
		free(hdata);
		return err;
	}

	/* Calculate memory bandwidth */
	bandwidth =
	    ((double) size * (double) ITERATIONS) / (time *
						     (double) (1 << 20));

	printf("%u %f\n", (unsigned int) size, bandwidth);

	/* Release memory objects */
	clReleaseMemObject(ddest);
	clReleaseMemObject(dsource);
	free(hdata);

	return err;
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
