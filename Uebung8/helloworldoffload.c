#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
 
int main(void)
{
    #pragma offload target(mic)
    int i;
    omp_set_num_threads(4);
 
    #pragma omp parallel for
    for (i = 0; i < 4; ++i)
    {
        const int id = omp_get_thread_num();
 
        printf("Hello World from thread %d\n", id);
 
        if (id == 0)
            /* Nur im Masterthread ausführen */
            printf("There are %d threads\n", omp_get_num_threads());
    }
 
    return EXIT_SUCCESS;
}
