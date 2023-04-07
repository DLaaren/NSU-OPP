#include <iostream>
#include <cmath>
#include <climits>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 2500
#define EPSILON 0.01 //10^(-5)
#define PARAMETER -0.1

float randDouble() {
    return ( (float)rand() / RAND_MAX ) * 3.0 - 2.0;
}

void generateMatrix(float *M, int columns, int rows) {
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            M[i*rows + j] = randDouble();
            //if you do not use tests
            /*if (i == j) {
                M[i*rows + j] = fabs(M[i*rows + j]) + 124.0; //выполнение условия диагонального преобладания
            }*/
        }
    }
}

void mul(float *M1, float *M2, float *resM, int size) {
    for (int i = 0; i < size; i++) {
        resM[i] = 0;
        for (unsigned int j = 0; j < N; j++) {
            resM[i] += M1[i*N + j] * M2[j];
        }
    }
}

void scalMult(float *M, float scalar, int size) {
    for (int i = 0; i < size; i++) {
        M[i] *= scalar;
    }
}

void sub(const float *M1, const float *M2, float *resM, int size) {
    for (int i = 0; i < size; i++) {
        resM[i] = M1[i] - M2[i];
    }
}

float preEuclideanNorm(const float* M, int size) {
    float norm = 0;
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

void printVector(float* vector, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("%lf ", vector[i]);
    }
    printf("\n");
}

void printMatrix(float* M, int columns, int rows) {
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
    srand(12345);

    if (MPI_Init(NULL,NULL) != MPI_SUCCESS) {
        perror("Error ocurred while trying to initiate MPI\n");
        exit(EXIT_FAILURE);
    }
    int processRank, sizeOfCluster;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    int* numberOfElementsVector = (int*)calloc(sizeOfCluster, sizeof(int));
    int* arrayOffsetsVector = (int*)calloc(sizeOfCluster, sizeof(int));
    int* numberOfElementsMatrix = (int*)calloc(sizeOfCluster, sizeof(int));
    int* arrayOffsetsMatrix = (int*)calloc(sizeOfCluster, sizeof(int));

    divideBetweenThreads(numberOfElementsVector, numberOfElementsMatrix, arrayOffsetsVector, arrayOffsetsMatrix, sizeOfCluster);

    int sizeOfBuffers = numberOfElementsVector[processRank];

    float* A_buffer = (float*)malloc(sizeof(float) * N * sizeOfBuffers);
    float* Ax_buffer = (float*)malloc(sizeof(float) * sizeOfBuffers);
    float* b_buffer = (float*)malloc(sizeof(float) * sizeOfBuffers);

    //all threads:
    float* A;
    float* b;
    float* prevX = (float*)malloc(sizeof(float) * N);
    float* nextX;
    float* Ax;
    float norm_b;
    float norm_Ax;
    float tau = PARAMETER;
    float result = INT_MAX;
    float lastResult = 0;
    unsigned int itCount = 0;

    //main thread:
    if (processRank == 0) {
        A = (float*)malloc(sizeof(float) * N * N);
        //if you do not use tests
        //generateMatrix(A, N, N);
        FILE* Abin = fopen("matA.bin", "r");
        if (Abin == NULL) {
            perror("fopen()");
            free(numberOfElementsVector);
            free(arrayOffsetsVector);
            free(numberOfElementsMatrix);
            free(arrayOffsetsMatrix);
            free(A_buffer);
            free(Ax_buffer);
            free(b_buffer);
            free(prevX);
            return -1;
        }
        fread(A, 4, N * N, Abin);
        fclose(Abin);

        Ax = (float*)malloc(sizeof(float) * N);

        b = (float*)malloc(sizeof(float) * N);
        //generateMatrix(b, N, 1);
        FILE* Bbin = fopen("vecB.bin", "r");
        if (Bbin == NULL) {
            perror("fopen()");
            free(numberOfElementsVector);
            free(arrayOffsetsVector);
            free(numberOfElementsMatrix);
            free(arrayOffsetsMatrix);
            free(A_buffer);
            free(Ax_buffer);
            free(b_buffer);
            free(prevX);
            return -1;
        }
        fread(b, 4, N, Bbin);
        fclose(Bbin);

        nextX = (float*)malloc(sizeof(float) * N);
        generateMatrix(nextX, N, 1);

        norm_b = preEuclideanNorm(b, N);
        norm_b = sqrt(norm_b); // ||b||
    }

    MPI_Bcast(prevX, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, numberOfElementsVector, arrayOffsetsVector, MPI_FLOAT, b_buffer, sizeOfBuffers, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A, numberOfElementsMatrix, arrayOffsetsMatrix, MPI_FLOAT, A_buffer, N * sizeOfBuffers, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float start = MPI_Wtime();
    while (result > EPSILON) {

        if (processRank == 0) {
            copyVector<float>(nextX, prevX, N);
        }
        
        MPI_Bcast(prevX, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

        mul(A_buffer, prevX, Ax_buffer, sizeOfBuffers); // Ax_buf = A_buf * prevX; 

        sub(Ax_buffer, b_buffer, Ax_buffer, sizeOfBuffers); // Ax_buf - b_buf

        float preNorm_Ax = preEuclideanNorm(Ax_buffer, sizeOfBuffers); // (Ax_buf - b_buf)^2

        MPI_Reduce(&preNorm_Ax, &norm_Ax, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); //adding all prenorms in the root

        scalMult(Ax_buffer, tau, sizeOfBuffers); // tau * (Ax_buf - b_buf)
        
        MPI_Gatherv(Ax_buffer, sizeOfBuffers, MPI_FLOAT, Ax, numberOfElementsVector, arrayOffsetsVector, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        if (processRank == 0) {
            sub(prevX, Ax, nextX, N); // prevX - tau * (Ax - b) = nextX

            norm_Ax = sqrt(norm_Ax);
            result = norm_Ax / norm_b; //  ||Ax - b|| / ||b||
            itCount++;

            if ((itCount >= 1000 && lastResult < result) || result == INFINITY) {
                if (tau > 0) {
                    printf("Doesn't have solution\n");
                    result = 0;
                } else {
                    printf("\nchanged tau\n");
                    tau = -PARAMETER;
                    result = INT_MAX;
                    itCount = 0;
                }
            }
            lastResult = result;
        }
        MPI_Bcast(&result, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); //for checking in the while
        //we don't need MPI_Barrier, because MPI_Bcast synchronize all processes
        //MPI_Barrier(MPI_COMM_WORLD); //synchronization for all threads
    }
    float end = MPI_Wtime();

    printf("time: %lf\n", end - start);

    float* answerX;
    if (processRank == 0) {
        answerX = (float*)malloc(sizeof(float) * N);
        FILE* Xbin = fopen("vecX.bin", "r");
        if (Xbin == NULL) {
            perror("fopen()");
            free(numberOfElementsVector);
            free(arrayOffsetsVector);
            free(numberOfElementsMatrix);
            free(arrayOffsetsMatrix);
            free(A_buffer);
            free(Ax_buffer);
            free(b_buffer);
            free(prevX);
            free(answerX);
            return -1;
        }
        fread(answerX, 4, N, Xbin);
        fclose(Xbin);

        FILE* resultX = fopen("result.bin", "w");
        if (resultX == NULL) {
            perror("fopen()");
            free(numberOfElementsVector);
            free(arrayOffsetsVector);
            free(numberOfElementsMatrix);
            free(arrayOffsetsMatrix);
            free(A_buffer);
            free(Ax_buffer);
            free(b_buffer);
            free(prevX);
            free(answerX);
            return -1;
        }
        fwrite(nextX, 4, N, resultX);
        fclose(resultX);
    }


    if (processRank == 0) {
        free(A);
        free(b);
        free(Ax);
        free(nextX);
        free(answerX);
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
        perror("Error ocurred while trying to close MPI\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}