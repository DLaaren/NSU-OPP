#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include "matrixOperations.h"

// A (N x M)     B (M x K)      C (N x K)

/*
     _ _ _ _ _+ Y
    |
    |
    |
    +
    X

*/

#define N 8
#define M 8
#define K 8
#define cluster_X 2
#define cluster_Y 2
#define NDIMS 2 //Number of dimensions of grid.

namespace Matrix {

static MPI_Comm gridComm;
static int gridCoords[NDIMS];

static MPI_Comm rowsComm;
static MPI_Comm columnsComm;

bool createGridComm() {
    //check if we can create such grid
    if ( (N % cluster_X != 0) && (K % cluster_Y != 0) ) {
        return false;
    }

    //create new communicators
    int dims[NDIMS] = {N, K}; //Integer array of size ndims specifying the number of processes in each dimension.
    int periods[NDIMS] = {1, 1}; //Logical array of size ndims specifying whether the grid is periodic (true) or not (false) in each dimension.

    //Makes a new communicator
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, 0, &gridComm);
    //Determines process coords
    MPI_Cart_coords(gridComm, 0, NDIMS, gridCoords);

    //The i-th entry of remain_dims specifies whether the i-th dimension is kept in the subgrid (true) or is dropped (false).
    int remain_dimsRows[NDIMS] = {1, 0};
    int remain_dimsColumns[NDIMS] = {0, 1};
    //create subgrids
    MPI_Cart_sub(gridComm, remain_dimsRows, &rowsComm);
    MPI_Cart_sub(gridComm, remain_dimsColumns, &columnsComm);

    return true;
}

void sendData(double* matrix_A, double* matrix_B, double* block_A, double* block_B, size_t blockSize_A, size_t blockSize_B) {
    //divide rows
    MPI_Scatter(matrix_A, N*M, MPI_DOUBLE, block_A, blockSize_A * M, MPI_DOUBLE, 0, rowsComm);
    MPI_Bcast(block_A, blockSize_A * M, MPI_DOUBLE, 0, columnsComm);

    //divide columns
    MPI_Datatype columnType, columnTypeResized;
    /*Defines a new data type that consists of a specified number of blocks of a specified size.
      Each block is a concatenation of the same number of elements of an existing data type.*/
    MPI_Type_vector(M, blockSize_B, K, MPI_DOUBLE, &columnType);

    //https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_commit.3.php
    MPI_Type_commit(&columnType);
    
    //https://rookiehpc.org/mpi/docs/mpi_type_create_resized/index.html
    MPI_Type_create_resized(columnType, 0, blockSize_B * sizeof(double), &columnTypeResized);
    MPI_Type_commit(&columnTypeResized);

    MPI_Scatter(matrix_B, 1, columnTypeResized, block_B, blockSize_B * M, MPI_DOUBLE, 0, columnsComm);
    MPI_Bcast(block_B, blockSize_B * M, MPI_DOUBLE, 0, rowsComm);
}

void collectData() {

}

int main() {
   if (MPI_Init(NULL,NULL) != MPI_SUCCESS) {
        perror("Error ocurred while trying to initiate MPI\n");
        exit(EXIT_FAILURE);
    }

    int processRank, sizeOfCluster;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    std::srand(123456);
    double* matrix_A;
    double* matrix_B;
    double* matrix_C;

    if (processRank == 0) {
        if (createGridComm() == false) {
            if (MPI_Finalize() != MPI_SUCCESS) {
                perror("Error ocurred while trying to close MPI\n");
                exit(EXIT_FAILURE);
            }
            return 0;
        }
        matrix_A = (double*)malloc(N*M * sizeof(double));
        matrix_B = (double*)malloc(M*K * sizeof(double));
        matrix_C = (double*)calloc(N*K, sizeof(double));
        generateMatrix(matrix_A, N, M);
        generateMatrix(matrix_B, M, K);
        setZero(matrix_C, N, K);
    }

    //create block for each proccess
    size_t blockSize_A = N / cluster_X;
    size_t blockSize_B = K / cluster_Y;
    size_t blockSize_C = blockSize_A * blockSize_B;
    double* block_A = (double*)malloc(blockSize_A * sizeof(double));
    double* block_B = (double*)malloc(blockSize_B * sizeof(double));
    double* block_C = (double*)calloc(blockSize_C, sizeof(double));


    double start = MPI_Wtime();

    sendData(matrix_A, matrix_B, block_A, block_B, blockSize_A, blockSize_B);

    simpleMultiplication(block_A, block_B, block_C, blockSize_A, M, blockSize_B);


    //MPI_Datatype smth;
    //MPI_Type_vector smth;
    collectData();

    double end = MPI_Wtime();
    printf("Time: %lf s\n", end - start);


    if (processRank == 0) {
        double* matrix_tmp = (double*)malloc(N*K * sizeof(double));
        simpleMultiplication(matrix_A, matrix_B, matrix_tmp, M, N, K);
        if (testMultiplication(matrix_C, matrix_tmp, N*K) == false) {
            printf("wrong parallel multiplication!\n");
        }
        free(matrix_tmp);
    }

    if (MPI_Finalize() != MPI_SUCCESS) {
        perror("Error ocurred while trying to close MPI\n");
        free(matrix_A);
        free(matrix_B);
        free(matrix_C);
        exit(EXIT_FAILURE);
    }

    free(matrix_A);
    free(matrix_B);
    free(matrix_C);
    return 0;
}

}