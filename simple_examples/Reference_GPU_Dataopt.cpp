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

    if (argc < 4 + size) {
        std::cerr << "not enough arguments: For " << size << " MPI processes you need at least " << 4 +size << " Arguments" << std::endl;
        exit(1);
    }
    
    int d1 = std::atoi(argv[1]);
    int d2 = std::atoi(argv[2]);
    int d3 = std::atoi(argv[3]);
    
    int iter = std::atoi(argv[rank + 4]);

    double * A = (double*) malloc(d1 * d2 * sizeof(double));
    double * B = (double*) malloc(d2 * d3 * sizeof(double));
    double * C = (double*) malloc(d1 * d3 * sizeof(double));
    
    for (int i = 0; i < d1 * d2; i++) {
        A[i] = 1;
    }
    for (int i = 0; i < d2 * d3; i++) {
        B[i] = 1;
    }
    for (int i = 0; i < d1 * d3; i++) {
        C[i] = 0;
    }

    for (int l = 0; l < omp_get_num_devices(); l++) {
        #pragma omp target enter data map(to:C[0:d1*d3]) map(to:A[0:d1*d2]) map(to:B[0:d2*d3]) map(to:d1,d2,d3) device(i)
    }

    double time = omp_get_wtime();   
    
    for (int l = 0; l < iter; l++) {
        #pragma omp target teams distribute parallel for device(l%omp_get_num_devices()) nowait
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    C[i * d3 + k] += A[i * d2 + j] * B[j * d3 + k];
                }   
            }
        }
    }
    
    #pragma omp taskwait

    MPI_Barrier(MPI_COMM_WORLD);
    time = omp_get_wtime() - time;

    for (int l = 0; l < omp_get_num_devices(); l++) {
        #pragma omp target exit data map(from:C[0:d1*d3]) map(delete:A[0:d1*d2]) map(delete:B[0:d2*d3]) map(delete:d1,d2,d3) device(i)
    }
    
    if (rank == 0)
    std::cout << "duration on process " << rank << ": " << time << std::endl;
    
    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
