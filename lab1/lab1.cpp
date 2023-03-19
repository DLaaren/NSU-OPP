
#include <iostream>
#include <cmath>
#include <climits>
#include <stdlib.h>
#include <mpi.h>

#define N 8
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

double preEuclideanNorm(const double *M, int size) {
    double norm = 0;
    for (int i = 0; i < size; i++) {
        norm += M[i]*M[i];
    }
    return norm;
}

void copyVector(const double *M1, double *M2) {
    for (int i = 0; i < N; i++) {
        M2[i] = M1[i];
    }
}

void printVector(double *vector) {
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("%lf ", vector[i]);
    }
    printf("\n");
}

void printMatrix(double *M, int columns, int rows) {
    printf("\n");
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            printf("%lf ", M[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    srand(time(NULL));

    if (MPI_Init(NULL,NULL) != MPI_SUCCESS) {
        printf("Error ocurred while trying to initiate MPI\n");
        exit(EXIT_FAILURE);
    }
    int processRank, sizeOfCluster;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    int numberOfColumnsPerThread = N / sizeOfCluster;
    if (N % sizeOfCluster != 0) {
        //what if N*N / sizeOfCluster has reminder???
    }

    double *A_buffer = (double*)malloc(sizeof(double) * N * numberOfColumnsPerThread);
    double *Ax_buffer = (double*)malloc(sizeof(double) * N * numberOfColumnsPerThread);
    double *b_buffer = (double*)malloc(sizeof(double) * numberOfColumnsPerThread);

    //all threads:
    double *A = NULL;
    double *b = NULL;
    double *prevX = (double*)malloc(sizeof(double) * N);
    double *nextX = (double*)malloc(sizeof(double) * N);
    double *Ax = (double*)malloc(sizeof(double) * N);
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

        b = (double*)malloc(sizeof(double) * N);
        generateMatrix(b, N, 1);

        generateMatrix(nextX, N, 1);

        norm_b = preEuclideanNorm(b, N);
        norm_b = sqrt(norm_b); // ||b||
    }

    MPI_Bcast(prevX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //divided the matrix A between all threads
    //https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-scatter-function
    //there is a garantee that number i thread will get number i peace of vector "b"
    MPI_Scatter(A, N * numberOfColumnsPerThread, MPI_DOUBLE, A_buffer, N * numberOfColumnsPerThread, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, numberOfColumnsPerThread, MPI_DOUBLE, b_buffer, numberOfColumnsPerThread, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start = MPI_Wtime();
    while (result > EPSILON) {

        if (processRank == 0) {
            copyVector(nextX, prevX);
        }

        MPI_Bcast(prevX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        mul(A_buffer, prevX, Ax_buffer, numberOfColumnsPerThread); // Ax_buf = A_buf * prevX; 

        sub(Ax_buffer, b_buffer, Ax_buffer, numberOfColumnsPerThread); // Ax_buf - b_buf

        double preNorm_Ax = preEuclideanNorm(Ax_buffer, numberOfColumnsPerThread); // (Ax_buf - b_buf)^2

        MPI_Reduce(&preNorm_Ax, &norm_Ax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); //adding all prenorms in the root

        scalMult(Ax_buffer, tau, numberOfColumnsPerThread); // tau * (Ax_buf - b_buf)

        MPI_Gather(Ax_buffer, numberOfColumnsPerThread, MPI_DOUBLE, Ax, numberOfColumnsPerThread, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    //printVector(nextX);

    free(A);
    free(b);
    free(prevX);
    free(Ax);
    free(nextX);

    if (MPI_Finalize() != MPI_SUCCESS) {
        printf("Error ocurred while trying to close MPI\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}