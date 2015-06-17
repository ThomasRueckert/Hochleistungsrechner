#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

typedef struct color_s
{
	char r;
	char g;
	char b;
} color_t;

void createBitmapHeader(FILE *outputFile, unsigned int width, unsigned int height)
{
	// String to identify the file as a BMP file - 4D42 hex
	char const bfType[] = "BM"; //FIXME

	// Offset address (in bytes) between the beginning of the file and the bitmap image data
	unsigned int bfOffBits = 54;

	// The size of the BMP file in bytes
	// Headersize + width * height * (r + g + b) * sizeof(char);
	unsigned int bfSize = bfOffBits + width * height * 3 * sizeof(char); 

	// Reserved for future usage
	unsigned int reserved = 0;

	// Size of the bitmap information header
	unsigned int biSize = 40;

	// Width of the bitmap in pixels
	unsigned int biWidth = width;

	// Height of the bitmap in pixels
	// negative = top-down, positive = bottom-up
	int biHeight = height;

	// Number of colorplanes. Usually 1 for BMP
	unsigned short biPlanes = 1;

	// Number of bits per pixel - 1,4,8 or 24
	unsigned short biBitCount = 24; //FIXME

	// Compression rate. 0 = uncompressed
	unsigned int biCompression = 0;

	// Size of image in pixels
	unsigned int biSizeImage = width * height;

	// Horizontal resolution
	unsigned int biXPelsPerMeter = 0x0b13;

	// Vertical resolution
	unsigned int biYPelsPerMeter = 0x0b13;

	// Number of colors in the color palette
	unsigned int biClrUsed = 0;

	// Number of important colors
	unsigned int biClrImportant = 0; //FIXME

	fprintf(outputFile,"%s",bfType);
	fwrite(&bfSize,4,1,outputFile);
	fwrite(&reserved,4,1,outputFile);
	fwrite(&bfOffBits,4,1,outputFile);

	fwrite(&biSize,4,1,outputFile);
	fwrite(&biWidth,4,1,outputFile);
	fwrite(&biHeight,4,1,outputFile);
	fwrite(&biPlanes,2,1,outputFile);
	fwrite(&biBitCount,2,1,outputFile);
	fwrite(&biCompression,4,1,outputFile);
	fwrite(&biSizeImage,4,1,outputFile);
	fwrite(&biXPelsPerMeter,4,1,outputFile);
	fwrite(&biYPelsPerMeter,4,1,outputFile);
	fwrite(&biClrUsed,4,1,outputFile);
	fwrite(&biClrImportant,4,1,outputFile);
}

color_t calculate(double x, double y)
{
	color_t color;
	double kt = 319, m = 4.0;
	double k = 0, wx = 0.0, wy = 0.0;
	double r;

	do
	{
		double tx = wx * wx - (wy * wy + x);
		double ty = 2.0 * wx * wy + y;
		wx = tx;
		wy = ty;
		r = wx * wx + wy * wy;
		k = k + 1;
	}
	while((r<=m) && (k<=kt));

	if (k>kt) 
	{
		// Black
		color.r = 0;
		color.g = 0;
		color.b = 0;
	}		
	else if (k<16) 
	{ 
		color.r = (char)(k * 8); 
		color.g = (char)(k * 8); 
		color.b = (char)(128 + k * 4); 
	}
	else if (k>=16 && k<64) 
	{ 
		color.r = (char)(128 + k - 16); 
		color.g = (char)(128 + k - 16); 
		color.b = (char)(192 + k - 16);
	}
	else if (k>=64) 
	{ 
		color.r = (char)(kt - k); 
		color.g = (char)(128 + (kt - k) / 2); 
		color.b = (char)(kt - k);
	}

	return color;
}

int main()
{

  
	char const outputFileName[] = "output.bmp";

	unsigned int width = 10000;
	unsigned int height = 10000;

	FILE *outputFile = fopen(outputFileName,"w");

	createBitmapHeader(outputFile, width, height);

	double xmin = 2.1, xmax = -0.6, ymin = -1.35, ymax = 1.35;
	double dx = (xmax - xmin) / width ;
	double dy = (ymax - ymin) / height; 
	
	#pragma omp parallel default(none) for ordered
	for (unsigned int i = 0; i < height; i++)
	{
		double jy = ymin + i * dy;
		#pragma omp parallel default(none) for ordered
		for (unsigned int j = 0; j < width; j++)
		{
			double jx = xmin + j * dx;

			color_t color = calculate(jx, jy);

			
			fwrite(&color.b, 1, 1, outputFile);
			fwrite(&color.g, 1, 1, outputFile);
			fwrite(&color.r, 1, 1, outputFile);
		}		
	}

	fclose(outputFile);

	//return 0;
}
