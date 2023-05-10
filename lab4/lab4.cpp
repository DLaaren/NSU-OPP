#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cfloat>
#include <cstring>

#define N 100
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

void printCube(double* M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                    printf(" %7.4lf", M[point(i,j,k)]);
            }
            printf(";");
        }
        printf("\n");
    }
}

double get_Xi(int i) {
    return Xo + i * Hx;
}

double get_Yj(int j) {
    return Yo + j * Hy;
}

double get_Zk(int k) {
    return Zo + k * Hz;
}

double fi_function(double x, double y, double z) {
    return x*x + y*y + z*z;
}

double ro_fucntion(double x, double y, double z) {
    return 6 - a * fi_function(x, y, z);
}

//шаг 1-2 :: задать значения искомой функции на границе и внутри области омега
void init_fi_function(int layerSize, double* currLayer) {
    int start_Z_for_each_process = processRank * layerSize - 1;
    for (int k = 0; k < layerSize + 2; k++) {
        double z = get_Zk(k + start_Z_for_each_process);     

        for (int i = 0; i < N; i++) {
            double x = get_Xi(i);

            for (int j = 0; j < N; j++) {
                double y = get_Yj(j);

                if (k != 0 && k != N-1 &&
                    j != 0 && j != N-1 &&
                    i != 0 && i != N-1) {
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

double updateLayer() {

}

int main() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    if (N % commSize && processRank == 0) {
        std::cout << "Invalid grid size\n";
        return 0;
    }

    int layerSize = N / commSize;
    int layerCoord_indexZ = processRank * layerSize + Zo;
    //объем больше одного блока, т к будем брать краи соседей --> +2 (сверху и снизку по +1)
    int layerVolume = (layerSize + 2) * N * N;
    double* currLayer = new double[layerVolume];
    double* currLayerBuf = new double[layerVolume];

    MPI_Request req[4];

    double start = MPI_Wtime();

    init_fi_function(layerSize, currLayer);

    double globalDelta = DBL_MAX;
    do {
        //send and receive in background
        if (processRank != 0) {
            //lower part of layer
            MPI_Isend(currLayerBuf + N*N, N*N, MPI_DOUBLE, processRank - 1, 123, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(currLayerBuf, N*N, MPI_DOUBLE, processRank - 1, 123, MPI_COMM_WORLD, &req[1]);
        }

        if (processRank != commSize - 1) {
            //upper part of layer
            MPI_Isend(currLayerBuf + layerSize * N*N, N*N, MPI_DOUBLE, processRank + 1, 123, MPI_COMM_WORLD, &req[2]);
            MPI_Irecv(currLayerBuf + (layerSize + 1) * N*N, N*N, MPI_DOUBLE, processRank + 1, 123, MPI_COMM_WORLD, &req[3]);
        }

        double localDelta = DBL_MIN;
        double tmpDelta;

        //update layers
        for () {
            updateLayer();
        }

        //wait for upper and lower parts
        //MPI_Wait()

        //update these parts

        //gatherAll
        MPI_Allreduce(&localDelta, &globalDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    } while (globalDelta > epsilon);

    double* omega;
    //gather it all back
    if (processRank == 0) {
        omega = new double[N * N * N];
    }

    MPI_Gather();

    double end = MPI_Wtime();

    if (processRank == 0) {
        std::cout << "Time: " << end - start << "\n";
    }

    if (processRank == 0) {
        printCube(omega);
        delete [] omega;
    }

    delete [] currLayer;
    delete [] currLayerBuf;

    MPI_Finalize();
    return 0;
}