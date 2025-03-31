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

    // enable pinned memory only when PINNED is explcitly enabled
    bool pinned = false;
    if (auto *pinned_env = std::getenv("PINNED")) {
        std::string pinned_str = pinned_env;
        pinned = !(pinned_str == "0" || pinned_str.empty());
    }

    double *A, *B, *C;

    if (pinned) {
        A = (double*)omp_alloc(d1 * d2 * sizeof(double), llvm_omp_target_host_mem_alloc);
        B = (double*)omp_alloc(d2 * d3 * sizeof(double), llvm_omp_target_host_mem_alloc);
        C = (double*)omp_alloc(d1 * d3 * sizeof(double), llvm_omp_target_host_mem_alloc);
    } else {
        A = (double*) malloc(d1 * d2 * sizeof(double));
        B = (double*) malloc(d2 * d3 * sizeof(double));
        C = (double*) malloc(d1 * d3 * sizeof(double));
    }

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
    
    for (int l = 0; l < iter; l++) {
        #pragma omp target teams distribute parallel for map(from:C[0:d1*d3]) map(to:A[0:d1*d2]) map(to:B[0:d2*d3]) device(device) collapse(2) nowait
        for (int i = 0; i < d1; i++) {
            for (int k = 0; k < d3; k++) {
                C[i * d3 + k] = 0;
                for (int j = 0; j < d2; j++) {
                    C[i * d3 + k] += A[i * d2 + j] * B[j * d3 + k];
                }   
            }
        }
    }
    
    #pragma omp taskwait

    MPI_Barrier(MPI_COMM_WORLD);
    time = omp_get_wtime() - time;
    
    if (rank == 0) {
        std::cout << "duration on process " << rank << ": " << time << std::endl;
        //std::cout << "Result:  " << C[0] << std::endl;
    }

    if (pinned) {
        omp_free(A, llvm_omp_target_host_mem_alloc);
        omp_free(B, llvm_omp_target_host_mem_alloc);
        omp_free(C, llvm_omp_target_host_mem_alloc);
    } else {
        free(A);
        free(B);
        free(C);
    }

    //finalizeTargetDART();
    MPI_Finalize();
    return 0;
}
