@g++ -std=c++17 -O3 -ffast-math -mavx2 -mfma sgemm2.cpp -c -o sgemm2.o
@g++ -std=c++17 -O3 -fopenmp -ffast-math -DMKL -DEXEC -mavx2 -mfma ../../OpenBLAS/bin/libopenblas.dll sgemm2.cpp -o sgemm2.exe