all: hr_u3_a1_sobel

hr_u3_a1_sobel: hr_u3_a1_sobel.c
	gcc -Wall `MagickWand-config --cflags --cppflags` -lOpenCL -o $@ $< `MagickWand-config --ldflags --libs`

clean:
	rm -f hr_u3_a1_sobel
