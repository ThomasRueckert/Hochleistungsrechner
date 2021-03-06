#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<immintrin.h>

#define HOST 	0
#define MIC 	1

__attribute__((target(mic)))
double calculateArea (double a, double b, unsigned long long N) 
{
	double area = 0;
	//custom
	double sum = 0;
 	double x1 = a;
 	double x2 = a;
	double y1 = 0;
 	double y2 = 0;

	area = ((b-a)/(2*N));
	double baN = (b-a)/N;
	
	#ifdef __MIC__
		
#pragma offload target(mic)		
#pragma omp parallel for schedule(dynamic, 20000) reduction(+:sum) private(x1,x2,y1,y2)
		for (unsigned long long i = 0; i < N; i++) {
		    
		    x1 = i*baN + a;
		    x2 = x1 + baN;
		    
		      y1 = pow(x1, 3) - 2*(pow(x1,2)) -x1+ 3;
		      y2 = pow(x2, 3) - 2*(pow(x2,2)) -x2+ 3;
		    
		    sum += (y1+y2);
		}

		area *= sum;
		// FIXME 3. u. 5. Aufgabe
	#else
		
#pragma omp parallel for schedule(dynamic, 10000) reduction(+:sum) private(x1,x2,y1,y2)
		for (unsigned long long i = 0; i < N; i++) {
		    
		    x1 = i*baN + a;
		    x2 = x1 + baN;
		    
		      y1 = pow(x1, 3) - 2*(x1*x1) -x1+ 3;
		      y2 = pow(x2, 3) - 2*(x2*x2) -x2+ 3;
		    
		    sum += (y1+y2);
		}

		area *= sum;
	#endif

	return area;
}

int main() 
{
	unsigned long long N = (10000000000 / 8) * 8; // 10.000.000.000
	double a = -1.0;
	double b = +2.8;
	double dx = (b - a) / (double) N;
	double area = -1;
	int where = HOST;

	printf("a: \t %lf\n", a);
	printf("b: \t %lf\n", b);
	printf("dx: \t %.14lf\n", dx);

	double startTime = omp_get_wtime();

	#pragma offload target(mic) in(a, b, N) out(area, where) // FIXME mic:<id>
	{
		#ifdef __MIC__
			where = MIC;
		#else
			where = HOST;
		#endif

		area = calculateArea(a, b, N);
	}

	double endTime = omp_get_wtime();

	printf("area: \t %.14lf\n", area);
	printf("time: \t %.14lf\n", endTime - startTime);
	printf("where: \t ");

	if (where == MIC) 
		printf("MIC\n");
	else
		printf("HOST\n");
	
	return 0;
}
