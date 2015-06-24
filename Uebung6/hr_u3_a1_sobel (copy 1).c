/*
 *  Sobel image filter for OpenCL with support for multiple image formats
 *  Partly inspired by AMD and NVIDIA SDK examples
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
 *  Last modifications: oere, 2011-06-07, 14:50
 *
 *  astyle --style=k/r -t8
 */

#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>		/* ENOMEM */
#include <sys/time.h>		/* gettimeofday() */

#include <wand/MagickWand.h>	/* ImageMagick for image conversion (libmagickwand-dev) */

#define DEBUG 1

/* OpenCL */
const char oclKernelName[] = "sobelFilter";
const char *oclKernelSource[] = {
	"__kernel void sobelFilter(const __global uchar4 *inputImage, __global uchar4 *outputImage)\n"
	"{\n"
	"	float4 Gx = 0;\n"
	"	float4 Gy = 0;\n"
	"	const int posx = get_global_id(0);\n"
	"	const int posy = get_global_id(1);\n"
	"	const int width = get_global_size(0);\n"
	"	const int height = get_global_size(1);\n"
	"\n"
	"	const int pos = posx + posy * width;\n"
	"\n"
	"	/* Handle edge cases - maybe ignore */\n"
	"	if (posx == 0 || posx == width-1 || posy == 0 || posy == height-1) {\n"
	"		outputImage[pos] = inputImage[pos];\n"
	"	} else {\n"
	"		float4 i00 = convert_float4(inputImage[pos - 1 - width]);\n"
	"		float4 i01 = convert_float4(inputImage[pos - 0 - width]);\n"
	"		float4 i02 = convert_float4(inputImage[pos + 1 - width]);\n"
	"		float4 i10 = convert_float4(inputImage[pos - 1 - 0]);\n"
	"		float4 i11 = convert_float4(inputImage[pos - 0 - 0]);\n"
	"		float4 i12 = convert_float4(inputImage[pos + 1 - 0]);\n"
	"		float4 i20 = convert_float4(inputImage[pos - 1 + width]);\n"
	"		float4 i21 = convert_float4(inputImage[pos - 0 + width]);\n"
	"		float4 i22 = convert_float4(inputImage[pos + 1 + width]);\n"
	"\n"
	"		Gx = i00 + (float4)(2) * i10 + i20 - i02 - (float4)(2) * i12 - i22;\n"
	"		Gy = i00 - i20 + (float4)(2)*i01 - (float4)(2)*i21 + i02 - i22;\n"
	"\n"
	"		outputImage[pos] = convert_uchar4(native_sqrt(Gx*Gx + Gy*Gy));\n"
	"	}\n"
	"\n"
	"}\n"
};
const char oclBuildOpts[] = { "-Werror" };

cl_int showDeviceInfo(cl_device_id *);

/* ImageMagick */
struct image {
	char *path; /* input/output path of file to read/write */
	unsigned char *data; /* pointer to RGB data */
	size_t dataLength; /* length of the RGB data */
	size_t width; /* image width */
	size_t height; /* image height */
};
typedef struct image image_t;

int readImage(image_t *); /* Read image data from file; input = path */
int writeImage(image_t *); /* Write image data to file: input = path, width, height, data */
void deleteImage(image_t *);

/* Helper function */
double stopwatch(void);

int main(int argc, char *argv[]) {
	int err = 0;
	char oclBuildLog[1024];
	image_t inputImage = { NULL, NULL, 0, 0, 0 };
	image_t outputImage = { NULL, NULL, 0, 0, 0 };

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
	cl_mem oclInputImage, oclOutputImage;
	size_t oclOutputImageSize = -1;
	size_t oclGlobalWorkSize[2] = { 0, 0 };

	/* Check for arguments */
	if (argc < 3) {
		printf("Usage: %s <path-to-input-image> <path-to-output-image>\n"\
		       "with support for major extensions like jpg, bmp, gif, ...\n", argv[0]);
		return -1;
	}

	/* Read/import input image through ImageMagick */
	inputImage.path = argv[1];
	err = readImage(&inputImage);
#if DEBUG
	printf("inputImage\n\tpath = %s\n\tdata = %p\n\tdataLength = %zu\n\twidth = %zu\n\theight = %zu\n",
	       inputImage.path, inputImage.data, inputImage.dataLength, inputImage.width, inputImage.height);
#endif
	if (err) {
		printf("%s:%d - readImage('%s') failed - Exiting...\n",__FILE__, __LINE__, inputImage.path);
		return -1;
	}

	/* Configure output image */
	outputImage.path = argv[2];
	outputImage.width = inputImage.width;
	outputImage.height = inputImage.height;

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
		("%s:%d - clGetProgramBuildInfo() returned %d - CL_PROGRAM_BUILD_LOG:\n'%s'\n",
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
	oclInputImage = clCreateBuffer(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputImage.dataLength, inputImage.data, &oclErr);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clCreateBuffer() for oclInputImage returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	oclOutputImage = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, inputImage.dataLength, NULL, &oclErr);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clCreateBuffer() for oclOutputImage returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Set kernel arguments */
	oclErr = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), &oclInputImage);
	oclErr |= clSetKernelArg(oclKernel, 1, sizeof(cl_mem), &oclOutputImage);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clSetKernelArg() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Execute the kernel two dimensional */
	oclGlobalWorkSize[0] = outputImage.width;
	oclGlobalWorkSize[1] = outputImage.height;
	oclErr = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, oclGlobalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clEnqueueNDRangeKernel() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Calculate required output image size */
	oclErr = clGetMemObjectInfo(oclOutputImage, CL_MEM_SIZE, sizeof(oclOutputImageSize), &oclOutputImageSize, NULL);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clGetMemObjectInfo() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Allocate memory for output image */
	outputImage.data = (unsigned char *)malloc(oclOutputImageSize);
	if (NULL == outputImage.data) {
		printf
		("%s:%d - malloc() for outputImage.data failed - Exiting...\n",
		 __FILE__, __LINE__);
		return -ENOMEM;
	}
	outputImage.dataLength = oclOutputImageSize;

	/* Read the results back to the host memory */
	oclErr = clEnqueueReadBuffer(oclCommandQueue, oclOutputImage, CL_TRUE, 0, outputImage.dataLength, outputImage.data, 0, NULL, NULL);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - clEnqueueReadImage() returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	/* Write/export output image through ImageMagick */
	err = writeImage(&outputImage);
	if (err) {
		printf("%s:%d - writeImage('%s') failed - Exiting...\n",__FILE__, __LINE__, outputImage.path);
		return -1;
	}

	/* Clean up */
	deleteImage(&outputImage);
	deleteImage(&inputImage);

	oclErr = clReleaseKernel(oclKernel);
	oclErr |= clReleaseProgram(oclProgram);
	oclErr |= clReleaseCommandQueue(oclCommandQueue);
	oclErr |= clReleaseContext(oclContext);
	if (CL_SUCCESS != oclErr) {
		printf("%s:%d - Releasing OpenCL objects returned %d - failed!\n",
		       __FILE__, __LINE__, oclErr);
		return -1;
	}

	//printf("-- > %s:%d oclErr = %d < --\n", __FILE__, __LINE__, oclErr);
#if DEBUG
	printf("Exiting succesfully... \n");
#endif
	return 0;
}

/* Print device information */
cl_int showDeviceInfo(cl_device_id * oclDevice) {
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

int readImage(image_t *image) {
	MagickWand *magickWand;
	MagickBooleanType magickStatus;
	unsigned char * temp;

	MagickWandGenesis();
	magickWand = NewMagickWand();
	magickStatus = MagickReadImage(magickWand, image->path);
	if (MagickFalse == magickStatus) {
		printf("%s:%d - MagickReadImage(%s) failed!\n", __FILE__, __LINE__, image->path);
		return -1;
	}
	magickStatus = MagickSetImageDepth(magickWand, 8);
	if (MagickFalse == magickStatus) {
		printf("%s:%d - MagickSetImageDepth(8) failed!\n", __FILE__, __LINE__);
		return -1;
	}
	/* RGBA - Optimize for 128 bit OpenCL data handling */
	magickStatus = MagickSetImageFormat(magickWand, "RGBA");
	if (MagickFalse == magickStatus) {
		printf("%s:%d - MagickSetImageFormat(RGBA) failed!\n", __FILE__, __LINE__);
		return -1;
	}
	image->width = MagickGetImageWidth(magickWand);
	image->height = MagickGetImageHeight(magickWand);
	temp = MagickGetImageBlob(magickWand, &(image->dataLength));
	if (NULL == temp) {
		printf("%s:%d - MagickGetImageBlob() failed!\n", __FILE__, __LINE__);
		return -1;
	}
	image->data = (unsigned char *) malloc(image->dataLength);
	if (NULL == image->data) {
		printf("%s:%d - malloc() of image.data failed!\n", __FILE__, __LINE__);
		return -1;
	}
	memcpy(image->data, temp, image->dataLength);
	temp = MagickRelinquishMemory(temp);
	magickWand = DestroyMagickWand(magickWand);
	MagickWandTerminus();

	return 0;
}

int writeImage(image_t *image) {
	MagickWand *magickWand;
	MagickBooleanType magickStatus;

	MagickWandGenesis();
	magickWand = NewMagickWand();
	magickStatus = MagickConstituteImage(magickWand, image->width, image->height, "RGBA", CharPixel, image->data);
	if (MagickFalse == magickStatus) {
		printf("%s:%d - MagickConstituteImage() failed!\n", __FILE__, __LINE__);
		return -1;
	}
	magickStatus = MagickWriteImage(magickWand, image->path);
	if (MagickFalse == magickStatus) {
		printf("%s:%d - MagickWriteImage(%s) failed!\n", __FILE__, __LINE__, image->path);
		return -1;
	}

	magickWand = DestroyMagickWand(magickWand);
	MagickWandTerminus();
	return 0;
}

void deleteImage(image_t *image) {
	image->path = NULL; /* provided externally */
	if (NULL != image->data) {
		free(image->data);
		image->data = NULL;
	}
	image->dataLength = 0;
	image->width = 0;
	image->height = 0;
}

double stopwatch(void) {
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
