#include <stdlib.h>
#include <iostream>
#include <omp.h>
#define n 12800


typedef struct point{
    int x;
    int y[3];
} point_t;

int main(int argc, char** argv) {
	
    initTargetDART(&argc, &argv, (void *) &main);


    //std::cout << testFunction(&argc, &argv) << std::endl;

    int * points = (int*) malloc(n * sizeof(int));
    point_t point = {20, {1,2,3}};
    // reserve extra space to save the initial centroid placement
    int * memory = (int*) malloc(n * sizeof(int));
    int * memory2 = (int*) malloc(n * sizeof(int));                 

    std::cout << &point << std::endl;

    for (int i = 0; i < n; ++i) {
        points[i] = i;
    }

    omp_get_thread_num();

    int a = points[1];
    int b = points[1];

    std::cout << omp_get_num_devices() << std::endl;
    std::cout << omp_get_initial_device() << std::endl;

    std::cout << "point address" << points << std::endl;
    std::cout << "memory address" << memory << std::endl;

    # pragma omp target teams distribute parallel for map(from:memory[0:n]) device(100) nowait
        for (int i = 0; i < n; ++i) {
            memory[i] = point.x + a + point.y[2];
        }
    #pragma omp taskwait
    for (int j = 0; j < 4; j++) {
    # pragma omp target teams distribute parallel for map(from:memory2[0:n]) map(to:points[0:n]) device(100) nowait
        for (int i = 0; i < n; ++i) {
            memory2[i] = omp_get_device_num() + a - b;
        }
    }
    #pragma omp taskwait
	std::cout << memory[0] << std::endl;
    std::cout << memory2[0] << std::endl;
    free(points);
    free(memory);
    free(memory2);
    
    finalizeTargetDART();
    return 0;
}
