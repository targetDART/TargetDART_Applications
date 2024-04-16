#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>


int main(int argc, char** argv) {

    int rank, size;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	// targetDART initialization
    initTargetDART((void *) &main);

    if (argc < 4 + size) {
        std::cerr << "not enough arguments: For " << size << " MPI processes you need at least " << 5 +2*size << " Arguments" << std::endl;
        exit(1);
    }
    
    int chain = std::atoi(argv[1]);
    
    int iter = std::atoi(argv[2*rank + 2]);
    
    int d1 = std::atoi(argv[2*rank + 2 + 1]);
    int d2 = std::atoi(argv[2*rank + 2 + 2]);
    int d3 = std::atoi(argv[2*rank + 2 + 3]);

    int* pipe = (int*) malloc(iter * sizeof(int));
    double * A = (double*) malloc(d1 * d2 * sizeof(double));
    double * B = (double*) malloc(d2 * d3 * sizeof(double));
    double * C = (double*) malloc(d1 * d3 * sizeof(double));
    
    double time = omp_get_wtime();   
    
    #pragma omp parallel
    for (int p = 0; p < chain; p++) {
        #pragma omp single
        #pragma omp task depend(inout:chain)
        td_phase_progress(p);
        #pragma omp for
        for (int l = 0; l < iter; l++) {
            #pragma omp target teams distribute map(from:C[0:d1*d3]) map(to:A[0:d1*d2]) map(to:B[0:d2*d3]) map(to:d1,d2,d3) device(TARGETDART_ANY) nowait depend(inout:pipe[p]) depend(in:chain)
            for (int i = 0; i < d1; i++) {
                #pragma omp parallel for
                for (int k = 0; k < d3; k++) {
                    for (int j = 0; j < d2; j++) {
                        C[i * d3 + k] += A[i * d2 + j] * B[j * d3 + k];
                    }   
                }
            }
        }
        #pragma omp single
        #pragma omp task depend(inout:chain)
        td_advance(1);
    }
    
    #pragma omp taskwait
    time = omp_get_wtime() - time;
    
    std::cout << "duration on process " << rank << ": " << time << std::endl;
    
    free(A);
    free(B);
    free(C);
    
    finalizeTargetDART();
    MPI_Finalize();
    return 0;
}
