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
 	double x1[8];
 	double x2[8] = a;
	double y1[8] = 0;
 	double y2[8] = 0;

	area = ((b-a)/(2*N));
	double baN = (b-a)/N;
	
	#ifdef __MIC__
		
#pragma omp parallel for schedule(dynamic, 10000) reduction(+:sum) private(x1,x2,y1,y2)
		for (unsigned long long i = 0; i < N; i+= 8) {
		    
		    __m512d x1m512 = _mm512_set1_pd(baN);
		    __m512d am512 = _mm512_set1_pd(a);
		    for (int j = 0; j < 8; j++) {
		     x1[j] = i+j;
		    }
		    __m512d ij = _mm512_set_pd(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
		    
// 		    __m512d x2m512 = _mm512_set1_pd(baN);
		    
		    __m512d x1m512result = _mm512d_fmadd_pd(x1m512, ij, am512);
		    __m512d x2m512result = _mm512d_add_pd(x1m512result, baN);
		    
		    __m512d twom512 = _mm512_set1_pd(2);
		    __m512d threem512 = _mm512_set1_pd(3);
		    
		    
		    //y1
		    __m512d pow11 = _mm512_pow_pd(x1m512result, threem512);
		    __m512d pow12 = _mm512_pow_pd(x1m512result, twom512);
		    __m512d pow12mul2 = _mm512d_mul_pd(pow12, twom512);
		    
		    __m512d y1restmp1 = _mm512d_add_pd(threem512, pow11);
		    __m512d y1restmp2 =  _mm512d_sub_pd(y1restmp1, pow12mul2);
		    __m512d y1res =  _mm512d_sub_pd(y1restmp2, x1m512result);
		    
		    
		    //y2
		    __m512d pow21 = _mm512_pow_pd(x2m512result, threem512);
		    __m512d pow22 = _mm512_pow_pd(x2m512result, twom512);
		    __m512d pow22mul2 = _mm512d_mul_pd(pow22, twom512);
		    
		    __m512d y2restmp1 = _mm512d_add_pd(threem512, pow21);
		    __m512d y2restmp2 =  _mm512d_sub_pd(y2restmp1, pow22mul2);
		    __m512d y2res =  _mm512d_sub_pd(y2restmp2, x2m512result);
		    
		    //sum		    sum += (y1+y2);
		   __m512d res = _mm512d_add_pd(y1res, y2res);
		   __m512d sum512 = _mm512d_add_pd(res, sum512);
		    
		   sum += _mm512d_reduce_add_pd(sum512);
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
