#include <iostream>
#include <cmath>
#include <climits>
#include <stdlib.h>
#include <mpi.h>

#define N 1200
#define EPSILON 0.00001 //10^(-5)
#define PARAMETER 0.01

double randDouble() {
    return ( (double)rand() / RAND_MAX ) * 4.0 - 2.0;
}

void generateMatrix(double *M, int columns, int rows) {
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            M[i*rows + j] = randDouble();
            if (i == j) {
                M[i*rows + j] = fabs(M[i*rows + j]) + 124.0; //выполнение условия диагонального преобладания
            }
        }
    }
}

void mul(double *M1, double *M2, double *resM, int size) {
    for (int i = 0; i < size; i++) {
        resM[i] = 0;
        for (unsigned int j = 0; j < N; j++) {
            resM[i] += M1[i*N + j] * M2[j];
        }
    }
}

void scalMult(double *M, double scalar, int size) {
    for (int i = 0; i < size; i++) {
        M[i] *= scalar;
    }
}

void sub(const double *M1, const double *M2, double *resM, int size) {
    for (int i = 0; i < size; i++) {
        resM[i] = M1[i] - M2[i];
    }
}

double preEuclideanNorm(const double* M, int size) {
    double norm = 0;
    for (int i = 0; i < size; i++) {
        norm += M[i]*M[i];
    }
    return norm;
}

template <class T>
void copyVector(const T* M1, T* M2, int size) {
    for (int i = 0; i < size; i++) {
        M2[i] = M1[i];
    }
}

void printVector(double* vector, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("%lf ", vector[i]);
    }
    printf("\n");
}

void printMatrix(double* M, int columns, int rows) {
    printf("\n");
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            printf("%lf ", M[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void divideBetweenThreads(int* numberOfElementsVector, int* numberOfElementsMatrix, int* arrayOffsetsVector, int* arrayOffsetsMatrix, int sizeOfCluster) {
    int reminder = N % sizeOfCluster;
    int lastOffset = 0;
    if (reminder != 0) {
        for (int i = 0; i < reminder; i++) {
            numberOfElementsVector[i] = 1;
        }
    }
    for (int i = 0; i < sizeOfCluster; i++) {
        numberOfElementsVector[i] += N / sizeOfCluster;
        arrayOffsetsVector[i] = lastOffset;
        lastOffset += numberOfElementsVector[i];
    }

    copyVector<int>(numberOfElementsVector, numberOfElementsMatrix, sizeOfCluster);

    lastOffset = 0;
    for (int i = 0; i < sizeOfCluster; i++) {
        numberOfElementsMatrix[i] *= N;
        arrayOffsetsMatrix[i] = lastOffset;
        lastOffset += numberOfElementsMatrix[i];
    }
}

int main() {
    srand(123456);

    if (MPI_Init(NULL,NULL) != MPI_SUCCESS) {
        printf("Error ocurred while trying to initiate MPI\n");
        exit(EXIT_FAILURE);
    }
    int processRank, sizeOfCluster;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    int* numberOfElementsVector = (int*)calloc(sizeOfCluster, sizeof(int));
    int* arrayOffsetsVector = (int*)calloc(sizeOfCluster, sizeof(int));
    int* numberOfElementsMatrix = (int*)calloc(sizeOfCluster, sizeof(int));
    int* arrayOffsetsMatrix = (int*)calloc(sizeOfCluster, sizeof(int));

    if (processRank == 0) {
        divideBetweenThreads(numberOfElementsVector, numberOfElementsMatrix, arrayOffsetsVector, arrayOffsetsMatrix, sizeOfCluster);
    }

    MPI_Bcast(numberOfElementsVector, sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(arrayOffsetsVector, sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(numberOfElementsMatrix, sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(arrayOffsetsMatrix, sizeOfCluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int sizeOfBuffers = numberOfElementsVector[processRank];

    double* A_buffer = (double*)malloc(sizeof(double) * N * sizeOfBuffers);
    double* Ax_buffer = (double*)malloc(sizeof(double) * sizeOfBuffers);
    double* b_buffer = (double*)malloc(sizeof(double) * sizeOfBuffers);

    //all threads:
    double* A;
    double* b;
    double* prevX = (double*)malloc(sizeof(double) * N);
    double* nextX;
    double* Ax;
    double norm_b;
    double norm_Ax;
    double tau = PARAMETER;
    double result = INT_MAX;
    double lastResult = INT_MAX;
    unsigned int itCount = 0;

    //main thread:
    if (processRank == 0) {
        A = (double*)malloc(sizeof(double) * N * N);
        generateMatrix(A, N, N);

        Ax = (double*)malloc(sizeof(double) * N);

        b = (double*)malloc(sizeof(double) * N);
        generateMatrix(b, N, 1);

        nextX = (double*)malloc(sizeof(double) * N);
        generateMatrix(nextX, N, 1);

        norm_b = preEuclideanNorm(b, N);
        norm_b = sqrt(norm_b); // ||b||
    }
    MPI_Bcast(prevX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, numberOfElementsVector, arrayOffsetsVector, MPI_DOUBLE, b_buffer, sizeOfBuffers, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A, numberOfElementsMatrix, arrayOffsetsMatrix, MPI_DOUBLE, A_buffer, N * sizeOfBuffers, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start = MPI_Wtime();
    while (result > EPSILON) {

        if (processRank == 0) {
            copyVector<double>(nextX, prevX, N);
        }
        
        MPI_Bcast(prevX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        mul(A_buffer, prevX, Ax_buffer, sizeOfBuffers); // Ax_buf = A_buf * prevX; 

        sub(Ax_buffer, b_buffer, Ax_buffer, sizeOfBuffers); // Ax_buf - b_buf

        double preNorm_Ax = preEuclideanNorm(Ax_buffer, sizeOfBuffers); // (Ax_buf - b_buf)^2

        MPI_Reduce(&preNorm_Ax, &norm_Ax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); //adding all prenorms in the root

        scalMult(Ax_buffer, tau, sizeOfBuffers); // tau * (Ax_buf - b_buf)
        
        MPI_Gatherv(Ax_buffer, sizeOfBuffers, MPI_DOUBLE, Ax, numberOfElementsVector, arrayOffsetsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (processRank == 0) {
            sub(prevX, Ax, nextX, N); // prevX - tau * (Ax - b) = nextX

            norm_Ax = sqrt(norm_Ax);
            result = norm_Ax / norm_b; //  ||Ax - b|| / ||b||
            itCount++;

            if ((itCount >= 1000 && lastResult > result) || result == INFINITY) {
                if (tau < 0) {
                    printf("Doesn't have solution\n");
                    result = 0;
                } else {
                    tau = -PARAMETER;
                    result = INT_MAX;
                    itCount = 0;
                }
            }
            lastResult = result;
        }
        MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); //for checking in the while
        MPI_Barrier(MPI_COMM_WORLD); //synchronization for all threads
    }
    double end = MPI_Wtime();

    printf("time: %lf\n", end - start);
    if (processRank == 0) {
        printVector(nextX, 10);
        //printMatrix(A, N, N);
        printf("\n");
    }

    if (processRank == 0) {
        free(A);
        free(b);
        free(Ax);
        free(nextX);
    }
    free(numberOfElementsVector);
    free(numberOfElementsMatrix);
    free(arrayOffsetsVector);
    free(arrayOffsetsMatrix);
    free(A_buffer);
    free(Ax_buffer);
    free(b_buffer);
    free(prevX);

    if (MPI_Finalize() != MPI_SUCCESS) {
        printf("Error ocurred while trying to close MPI\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}