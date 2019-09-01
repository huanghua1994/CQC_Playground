export I_MPI_PIN=0
export OMP_NUM_THREADS=1 

export NTHREADS=1
numactl -m 1 mpirun -np 64 ./mpi_omp_dsygv_test.x
echo "=============================================="

export NTHREADS=2
numactl -m 1 mpirun -np 64 ./mpi_omp_dsygv_test.x
echo "=============================================="

export NTHREADS=4
numactl -m 1 mpirun -np 64 ./mpi_omp_dsygv_test.x
echo "=============================================="

export NTHREADS=8
numactl -m 1 mpirun -np 64 ./mpi_omp_dsygv_test.x
echo "=============================================="

