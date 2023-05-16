#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cfloat>
#include <cstring>

using namespace std;

#define N 320
#define a 10e5
#define epsilon 10e-8

#define Xo -1 
#define Yo -1
#define Zo -1

#define Dx 2
#define Dy 2
#define Dz 2

static double Hx = Dx / (double)(N-1);
static double Hy = Dy / (double)(N-1);
static double Hz = Dz / (double)(N-1);

#define point(i,j,k) N*N*(i) + N*(j) + (k) 

static int commSize;
static int processRank;

double get_X(int i) {
    return Xo + i * Hx;
}

double get_Y(int j) {
    return Yo + j * Hy;
}

double get_Z(int k) {
    return Zo + k * Hz;
}

double fi_function(double x, double y, double z) {
    return x*x + y*y + z*z;
}

void printOmega(double* A){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf(" %7.4f", A[point(i,j,k)]);
            }
            cout << ";";
        }
        cout << endl;
    }
}

double delta(double* omega){
    auto delta = DBL_MIN;
    double x, y, z;
    for(int i = 0; i < N; i++){
        z = get_Z(i);
        for (int j = 0; j < N; j++){
            x = get_X(j);
            for(int k = 0; k < N; k++){
                y = get_Y(k);
                delta = max(delta, abs(omega[point(i,j,k)] - fi_function(x, y, z)));
            }
        }
    }
    return delta;
}

double ro_fucntion(double x, double y, double z) {
    return 6 - a * fi_function(x, y, z);
}

//шаг 1-2 :: задать значения искомой функции на границе и внутри области омега
void init_fi_function(int layerSize, double* currLayer) {
    int relative_Z_coord = processRank * layerSize - 1;
    for (int i = 0; i < layerSize + 2; i++) {
        double z = get_Z(i + relative_Z_coord);     

        for (int j = 0; j < N; j++) {
            double x = get_X(j);

            for (int k = 0; k < N; k++) {
                double y = get_Y(k);

                if (k != 0 && k != N-1 &&
                    j != 0 && j != N-1 &&
                    z != Zo && z != Zo + Dz) {
                        //inside
                        currLayer[point(i,j,k)] = 0;
                } else {
                    //border
                    currLayer[point(i,j,k)] = fi_function(x,y,z);                    
                }
                
            }
        }
    }
}


//шаг 3 :: обновляем значение функции фи
double updateLayer(int relative_Z_coord, int layerIndex, double* currLayer, double* currLayerBuf) {
    double x, y, z;
    int absolute_Z_coord = relative_Z_coord + layerIndex;
    auto delta = DBL_MIN;

    if (absolute_Z_coord == 0 || absolute_Z_coord == N-1) {
        //border parts -> do not update
        std::memcpy(currLayerBuf + layerIndex * N * N, currLayer + layerIndex * N * N, N * N * sizeof(double));
        delta = 0;
    } else {
        z = get_Z(absolute_Z_coord);
        
        for (int i = 0; i < N; i++) {
            x = get_X(i);

            for (int j = 0; j < N; j++) {
                y = get_Y(j);

                if (i == 0 || i == N-1 || j == 0 || j == N-1) {
                    //border points --> just copy
                    currLayerBuf[point(layerIndex, i, j)] = currLayer[point(layerIndex, i, j)];
                } else {
                    currLayerBuf[point(layerIndex, i, j)] =
                            ((currLayer[point(layerIndex+1, i, j)] + currLayer[point(layerIndex-1, i, j)]) / (Hz*Hz) +
                             (currLayer[point(layerIndex, i+1, j)] + currLayer[point(layerIndex, i-1, j)]) / (Hx*Hx) +
                             (currLayer[point(layerIndex, i, j+1)] + currLayer[point(layerIndex, i, j-1)]) / (Hy*Hy) -
                             ro_fucntion(x, y, z)) / (2/(Hx*Hx) + 2/(Hy*Hy) + 2/(Hz*Hz) + a);
                
                     if (abs(currLayerBuf[point(layerIndex, i, j)] - currLayer[point(layerIndex, i, j)]) > delta){
                        delta = currLayerBuf[point(layerIndex, i, j)] - currLayer[point(layerIndex, i, j)];
                    }
                }
            }
        }
    }
    return delta;
}

int waitForIrecv(MPI_Request* req) {
    int finishedReq = -1;
    int isAllReqCompleted = 1;

    if (commSize == 1) return -1;

    MPI_Testall(4, req, &isAllReqCompleted, MPI_STATUS_IGNORE);
    if (!isAllReqCompleted) {
        if (processRank == 0) {
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
            return -1;
        }
        if (processRank == commSize-1) {
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            return -1;
        }
        if (processRank != 0 && processRank != commSize - 1) {
            MPI_Waitany(4, req, &finishedReq, MPI_STATUS_IGNORE);   
            if (finishedReq == 2 || finishedReq == 3) {
                finishedReq = waitForIrecv(req);
            }
        }
    }

    return finishedReq;
}


int main() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    if (N % commSize != 0) {
        std::cout << "Invalid grid size\n";
        return 0;
    }

    auto globalDelta = DBL_MAX;

    int layerSize = N / commSize;
    int relative_Z_coord_forLayer = processRank * layerSize - 1;
    //объем больше одного блока, т к будем брать краи соседей --> +2 (сверху и снизку по +1)
    int layerVolume = (layerSize + 2) * N * N;
    double* currLayer = new double[layerVolume];
    double* currLayerBuf = new double[layerVolume];
    MPI_Request req[4];

    double start = MPI_Wtime();

    init_fi_function(layerSize, currLayer);
    do {
        //send and receive in background
        if (processRank != 0) {
            //lower border parts
            MPI_Isend(currLayerBuf + N*N, N*N, MPI_DOUBLE, processRank - 1, 123, MPI_COMM_WORLD, &req[2]);
            MPI_Irecv(currLayerBuf, N*N, MPI_DOUBLE, processRank - 1, 123, MPI_COMM_WORLD, &req[0]);
        }

        if (processRank != commSize - 1) {
            //upper border parts
            MPI_Isend(currLayerBuf + N*N*layerSize, N*N, MPI_DOUBLE, processRank + 1, 123, MPI_COMM_WORLD, &req[3]);
            MPI_Irecv(currLayerBuf + N*N*(layerSize+1), N*N, MPI_DOUBLE, processRank + 1, 123, MPI_COMM_WORLD, &req[1]);
        }

        auto localDelta = DBL_MIN;
        double tmpDelta;

        //update layers for each Z_k; k = 1 ... layerSize - 1;
        for (int indexLayer = 2; indexLayer < layerSize; indexLayer++) {
            tmpDelta = updateLayer(relative_Z_coord_forLayer, indexLayer, currLayer, currLayerBuf);
            localDelta = max(tmpDelta, localDelta);
        }

        //wait for upper and lower parts
        int finishedReq = waitForIrecv(req);
        //cout << finishedReq << endl;

        if (finishedReq == -1) {
            tmpDelta = updateLayer(relative_Z_coord_forLayer, 1, currLayer, currLayerBuf);
            localDelta = max(tmpDelta, localDelta);
            tmpDelta = updateLayer(relative_Z_coord_forLayer, layerSize, currLayer, currLayerBuf);
            localDelta = max(tmpDelta, localDelta);
        } else {
            if (finishedReq == 0) {
                //got lower part firstly
                //for Z_k = 1
                tmpDelta = updateLayer(relative_Z_coord_forLayer, 1, currLayer, currLayerBuf);
                localDelta = max(tmpDelta, localDelta);

                //then wait for the upper one
                MPI_Wait(&req[1], MPI_STATUS_IGNORE);
                
                //for Z_k = layerSize;
                tmpDelta = updateLayer(relative_Z_coord_forLayer, layerSize, currLayer, currLayerBuf);
                localDelta = max(tmpDelta, localDelta);

            } else if (finishedReq == 1) {
                //got upper part firstly
                //for Z_k = layerSize;
                tmpDelta = updateLayer(relative_Z_coord_forLayer, layerSize, currLayer, currLayerBuf);
                localDelta = max(tmpDelta, localDelta);

                //then wait for the lower one
                MPI_Wait(&req[0], MPI_STATUS_IGNORE);

                //for Z_k = 1
                tmpDelta = updateLayer(relative_Z_coord_forLayer, 1, currLayer, currLayerBuf);
                localDelta = max(tmpDelta, localDelta);
            }
        } 

        std::memcpy(currLayer, currLayerBuf, layerVolume * sizeof(double));

        //gatherAll
        MPI_Allreduce(&localDelta, &globalDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    } while (globalDelta > epsilon);

    double* omega;
    //gather it all back
    if (processRank == 0) {
        omega = new double[N * N * N];
    }

    // + N*N <--- do not forget about empty border parts
    MPI_Gather(currLayer + N*N, layerSize * N*N, MPI_DOUBLE, omega,
               layerSize * N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (processRank == 0) {
        cout << "Time: " << end - start << "\n";
        cout << "Delta: " << delta(omega) << "\n";
        //printOmega(omega);
        delete [] omega;
    }

    delete [] currLayer;
    delete [] currLayerBuf;

    MPI_Finalize();
    return 0;
}