#include <iostream>
#include <cstdlib>
#include <mpi.h>

template<class T>
void generateMatrix(T* matrix, size_t rows, size_t columns) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < columns; j++) {
            matrix[i*columns + j] = (std::rand() % 10) * 0.1;
        }
    }
}

void printMatrix(double* matrix, size_t n, size_t m) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            printf("%lf ", matrix[i*m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template<class T>
void simpleMultiplication(T* M1, T* M2, T* res, size_t n, size_t m, size_t k) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k; j++) {
            for (size_t q = 0; q < m; q++) {
                res[i*k + j] += M1[i*m + q] * M2[q*k + j];
            } 
        }
    }
}

template<class T>
bool testMultiplication(T* M1, T* M2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (M1[i] != M2[i]) return false;
    }
    return true;
}


// A (N x M)     B (M x K)      C (N x K)

/*
     _ _ _ _ _+ Y
    |
    |
    |
    +
    X
*/

#define N 80
#define M 80
#define K 80
#define CLUSTER_X 4
#define CLUSTER_Y 2
#define NDIMS 2 //Number of dimensions of grid.

static MPI_Comm gridComm;
static int gridCoords[NDIMS];
static int processRank, sizeOfCluster;

static MPI_Comm rowsComm;
static MPI_Comm columnsComm;

bool createGridComm() {
    //check if we can create such grid
    if ( ((N % CLUSTER_X != 0) && (K % CLUSTER_Y != 0)) || sizeOfCluster != CLUSTER_X * CLUSTER_Y) {
        return false;
    }

    //create new communicators
    int numberProcessesPerDim[NDIMS] = {CLUSTER_X, CLUSTER_Y}; //Integer array of size ndims specifying the number of processes in each dimension.
    int periods[NDIMS] = {1, 1}; //Logical array of size ndims specifying whether the grid is periodic (true) or not (false) in each dimension.

    //Makes a new communicator
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, numberProcessesPerDim, periods, 0, &gridComm);
    //Determines process coords
    MPI_Cart_coords(gridComm, processRank, NDIMS, gridCoords);

    //The i-th entry of remain_dims specifies whether the i-th dimension is kept in the subgrid (true) or is dropped (false).
    int remain_dimRows[NDIMS] = {1, 0};
    int remain_dimColumns[NDIMS] = {0, 1};
    MPI_Cart_sub(gridComm, remain_dimRows, &rowsComm);
    MPI_Cart_sub(gridComm, remain_dimColumns, &columnsComm);

    return true;
}

void sendData(double* matrix_A, double* matrix_B, double* block_A, double* block_B, size_t blockSize_A, size_t blockSize_B) {
    //divide rows
    if (gridCoords[1] == 0) MPI_Scatter(matrix_A, blockSize_A * M, MPI_DOUBLE, block_A, blockSize_A * M, MPI_DOUBLE, 0, rowsComm);
    MPI_Bcast(block_A, blockSize_A * M, MPI_DOUBLE, 0, columnsComm);

    //divide columns
    MPI_Datatype columnType, columnTypeResized;
    MPI_Type_vector(M, blockSize_B, K, MPI_DOUBLE, &columnType);
    MPI_Type_commit(&columnType);
    MPI_Type_create_resized(columnType, 0, blockSize_B * sizeof(double), &columnTypeResized);
    MPI_Type_commit(&columnTypeResized);

    if (gridCoords[0] == 0) MPI_Scatter(matrix_B, 1, columnTypeResized, block_B, blockSize_B * M, MPI_DOUBLE, 0, columnsComm);
    MPI_Bcast(block_B, blockSize_B * M, MPI_DOUBLE, 0, rowsComm);
}

void collectData(double* block_C, size_t blockSize_C, double* matrix_C, size_t blockSize_A, size_t blockSize_B) {
    MPI_Datatype recvBlockType, recvBlockTypeResized;
    MPI_Type_vector(blockSize_A, blockSize_B, K, MPI_DOUBLE, &recvBlockType);
    MPI_Type_commit(&recvBlockType);

    MPI_Type_create_resized(recvBlockType, 0, blockSize_B * sizeof(double), &recvBlockTypeResized);
    MPI_Type_commit(&recvBlockTypeResized);

    int* recvcounts = (int*)malloc(CLUSTER_X * CLUSTER_Y * sizeof(int)); //The number of elements that is received from each process.
    int* displs = (int*)malloc(CLUSTER_X * CLUSTER_Y * sizeof(int)); //The location, relative to the recvbuf parameter, of the data from each communicator process. 
    
    int offset = 0;
    for (size_t blockCount = 0; blockCount < CLUSTER_X * CLUSTER_Y; blockCount++) {
        if (blockCount % CLUSTER_Y == 0 && blockCount != 0) {
            offset += blockSize_A * CLUSTER_Y - (CLUSTER_Y - 1);
        } else if (blockCount != 0) {
            offset += 1;
        }
        recvcounts[blockCount] = 1;
        displs[blockCount] = offset;
    }

    MPI_Gatherv(block_C, blockSize_C, MPI_DOUBLE, matrix_C, recvcounts, displs, recvBlockTypeResized, 0, MPI_COMM_WORLD);
    free(recvcounts);
    free(displs);
}

int main() {
   if (MPI_Init(NULL,NULL) != MPI_SUCCESS) {
        perror("Error ocurred while trying to initiate MPI\n");
        exit(EXIT_FAILURE);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    std::srand(1);
    double* matrix_A;
    double* matrix_B;
    double* matrix_C;

    if (createGridComm() == false) {
        printf("GridComm can not be created\n");
        if (MPI_Finalize() != MPI_SUCCESS) {
            perror("Error ocurred while trying to close MPI\n");
        }
        exit(EXIT_FAILURE);
    }

    if (processRank == 0) {
        matrix_A = (double*)malloc(N*M * sizeof(double));
        matrix_B = (double*)malloc(M*K * sizeof(double));
        matrix_C = (double*)calloc(N*K, sizeof(double));
        generateMatrix(matrix_A, N, M);
        generateMatrix(matrix_B, M, K);
    }
    
    //create block for each proccess
    size_t blockSize_A = (N / CLUSTER_X);
    size_t blockSize_B = (K / CLUSTER_Y);
    size_t blockSize_C = blockSize_A * blockSize_B;
    double* block_A = (double*)malloc(blockSize_A * M * sizeof(double));
    double* block_B = (double*)malloc(blockSize_B * M * sizeof(double));
    double* block_C = (double*)calloc(blockSize_C, sizeof(double));

    double start = MPI_Wtime();

    sendData(matrix_A, matrix_B, block_A, block_B, blockSize_A, blockSize_B);

    simpleMultiplication<double>(block_A, block_B, block_C, blockSize_A, M, blockSize_B);

    collectData(block_C, blockSize_C, matrix_C, blockSize_A, blockSize_B);

    double end = MPI_Wtime();

    if (processRank == 0) {
        double* matrix_tmp = (double*)calloc(N * K, sizeof(double));
        simpleMultiplication<double>(matrix_A, matrix_B, matrix_tmp, N, M, K);

        if (testMultiplication<double>(matrix_C, matrix_tmp, N * K) == false) {
            printf("wrong parallel multiplication!\n");
        } else {
            printf("Time: %lf s\n", end - start);
        }
        //printMatrix(matrix_tmp, N, K);
        //printf("\n\n");
        //printMatrix(matrix_C, N, K);
        free(matrix_tmp);
    }

    if (processRank == 0) {
        free(matrix_A);
        free(matrix_B);
        free(matrix_C);
    }
    free(block_A);
    free(block_B);
    free(block_C);

    if (MPI_Finalize() != MPI_SUCCESS) {
        perror("Error ocurred while trying to close MPI\n");
    }
    return 0;
}
