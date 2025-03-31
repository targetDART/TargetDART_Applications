#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <string>


int main(int argc, char** argv) {

    int rank, size;
    int provided;
    int device;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	// targetDART initialization
    td_init((void *) &main);

    if (std::getenv("MODE") == NULL) {
        device = TARGETDART_ANY;
    } else {

        std::string mode = std::getenv("MODE");
        if (mode == "CPU") {
            device = TARGETDART_CPU;
        } else if (mode == "GPU") {
            device = TARGETDART_OFFLOAD;
        } else {
            device = TARGETDART_ANY;
        }
    }

    //std::cout << "MPI size, rank: " << size << ", " << rank << std::endl;

    if (argc < 4 + size) {
        std::cerr << "not enough arguments: For " << size << " MPI processes you need at least " << 4 + size << " Arguments" << std::endl;
        exit(1);
    }
    
    int d1 = std::atoi(argv[1]);
    int d2 = std::atoi(argv[2]);
    int d3 = std::atoi(argv[3]);
    
    int iter = std::atoi(argv[rank + 4]);

    // enable paged memory only when PAGED is explcitly enabled
    bool paged = false;
    if (auto *paged_env = std::getenv("PAGED")) {
        std::string paged_str = paged_env;
        paged = !(paged_str == "0" || paged_str.empty());
    }

    double *A, *B, *C;
    int *d;

    if (paged) {
        A = (double*) malloc(d1 * d2 * sizeof(double));
        B = (double*) malloc(d2 * d3 * sizeof(double));
        C = (double*) malloc(d1 * d3 * sizeof(double));
        d = (int*) malloc(3 * sizeof(int));
    } else {
        A = (double*)omp_alloc(d1 * d2 * sizeof(double), llvm_omp_target_host_mem_alloc);
        B = (double*)omp_alloc(d2 * d3 * sizeof(double), llvm_omp_target_host_mem_alloc);
        C = (double*)omp_alloc(d1 * d3 * sizeof(double), llvm_omp_target_host_mem_alloc);
        d = (int*)omp_alloc(3*sizeof(int), llvm_omp_target_host_mem_alloc);
    }

    d[0] = d1;
    d[1] = d2;
    d[2] = d3;

    for (int i = 0; i < d1 * d2; i++) {
        A[i] = 1;
    }
    for (int i = 0; i < d2 * d3; i++) {
        B[i] = 1;
    }
    for (int i = 0; i < d1 * d3; i++) {
        C[i] = 0;
    }
    
    double time = omp_get_wtime();   
    #pragma omp target data map(to:A[0:d1*d2]) map(to:B[0:d2*d3]) map(to:d[0:3]) device(device)
    {
    for (int l = 0; l < iter; l++) {
        #pragma omp target teams distribute parallel for map(from:C[0:d1*d3]) device(device) collapse(2) nowait
        for (int i = 0; i < d[0]; i++) {
            for (int k = 0; k < d[2]; k++) {
                C[i * d[2] + k] = 0;
                for (int j = 0; j < d[1]; j++) {
                    C[i * d[2] + k] += A[i * d[1] + j] * B[j * d[2] + k];
                }   
            }
        }
    }    
    #pragma omp taskwait
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time = omp_get_wtime() - time;
    
    if (rank == 0) {
        std::cout << "duration on process " << rank << ": " << time << std::endl;
        //std::cout << "Result:  " << C[0] << std::endl;
    }
    
    if (paged) {
        free(A);
        free(B);
        free(C);
        free(d);
    } else {
        omp_free(A, llvm_omp_target_host_mem_alloc);
        omp_free(B, llvm_omp_target_host_mem_alloc);
        omp_free(C, llvm_omp_target_host_mem_alloc);
        omp_free(d, llvm_omp_target_host_mem_alloc);
    }
    
    //finalizeTargetDART();
    MPI_Finalize();
    return 0;
}
