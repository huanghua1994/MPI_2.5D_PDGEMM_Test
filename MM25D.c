#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "mkl.h"

#define VERIFY

#include "utils.c"

// Factor nproc = c * nproc_ij * nproc_ij where c is a factor of nproc_ij
void get_nproc_ij_c(int argc, char **argv, int *_nproc, int *_my_rank, int *_c, int *_nproc_ij)
{
    int my_rank, nproc;
    int c, nproc_ij, succ = 1;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    if (argc > 2) 
    {
        c = atoi(argv[2]);
        nproc_ij = (int) sqrt(nproc / c);
    } else {
        nproc_ij = (int) sqrt(nproc / 2);
        c = nproc / (nproc_ij * nproc_ij);
    }
    
    if (nproc_ij * nproc_ij * c != nproc) 
    {
        if (my_rank == 0)
            fprintf(stderr, "[ERROR] nproc = %d, c = %d, cannot decide nproc_ij satisfy c * (nproc_ij)^2 = nproc!!\n", nproc, c);
        succ = 0;
    }
    
    if (c > nproc_ij) 
    {
        if (my_rank == 0)
            fprintf(stderr, "[ERROR] c = %d is larger than nproc_ij = %d !\n", c, nproc_ij);
        succ = 0;
    }
    
    if (nproc_ij % c != 0)
    {
        if (my_rank == 0)
            fprintf(stderr, "[ERROR] c = %d is not a factor of nproc_ij = %d !\n", c, nproc_ij);
        succ = 0;
    }
    
    if (succ == 0)
    {
        MPI_Finalize();
        exit(1);
    }
    
    *_nproc = nproc;
    *_my_rank = my_rank;
    *_c = c;
    *_nproc_ij = nproc_ij;
}

int main(int argc, char* argv[]) 
{
    int root = 0;
    MPI_Status status; 

    MPI_Init(&argc, &argv);
    
    int my_rank, nproc, c, nproc_ij;
    get_nproc_ij_c(argc, argv, &nproc, &my_rank, &c, &nproc_ij);

    int n = get_problem_size(argc, argv, nproc_ij, my_rank);
    int n_local = n / nproc_ij;
    int local_bs = n_local * n_local;
    if (my_rank == 0) printf("[MMM25D] nproc = %d, n = %d, nproc_ij = %d, c = %d\n", nproc, n, nproc_ij, c);

    //prepare for the cartesian topology
    MPI_Comm cartcomm;
    int dims[3] = {nproc_ij, nproc_ij, c};
    int periods[3] = {1,1,0}, reorder=0,coords[3];
    mpi_check(MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cartcomm));
    MPI_Comm_rank(cartcomm, &my_rank);
    MPI_Cart_coords(cartcomm, my_rank, 3, coords);

    //Split the cartcomm into rowcomm, colcomm
    MPI_Comm rowcomm, colcomm, depthcomm;
    MPI_Comm_split(cartcomm, coords[1]*nproc_ij+coords[2], coords[0], &colcomm); //i as rank
    MPI_Comm_split(cartcomm, coords[0]*nproc_ij+coords[2], coords[1], &rowcomm); //j as rank
    MPI_Comm_split(cartcomm, coords[0]*nproc_ij+coords[1], coords[2], &depthcomm); //k as rank

    MPI_Comm dim_ij_comm; //used for scatter data and collect result
    MPI_Comm_split(cartcomm, coords[2], coords[0]*nproc_ij+coords[1], &dim_ij_comm); //ij as rank

    double *M, *NT, *P;
    int root_coords[3] = {0,0,0}; //with zero first
    MPI_Cart_rank(cartcomm, root_coords, &root); //get the dst
    if (root == my_rank) {
        initial_matrix(&M, &NT, &P, n, n, n);
    }

    //now each process has a small portion of local
    /*The matrix size is p * p*/
    int i, j, k;
    double *A = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16);
    double *BT = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16); //j, k
    double *C = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16);
    double *C0 = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16);

    //Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int sendcounts[nproc_ij*nproc_ij];
    int displs[nproc_ij*nproc_ij]; //value in block (resized) count
    init_subarrtype(root, my_rank, n, nproc_ij, n_local, &subarrtype, sendcounts, displs);

    //Now A/BT local are ready accorss layer 0.
    scatter_data(root, my_rank, coords, n, nproc_ij, n_local,
            dim_ij_comm, sendcounts, displs, subarrtype,
            A, BT, M, NT);

    mpi_check(MPI_Barrier(cartcomm));
    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();
    //the main MMM25D algorithm part
    //The real start point.
    mpi_check(MPI_Bcast(A, n_local*n_local, MPI_DOUBLE, 0, depthcomm));
    mpi_check(MPI_Bcast(BT,n_local*n_local, MPI_DOUBLE, 0, depthcomm));

    //first need shift both A and B.
    int dst, src; //use for shift
    int datatag = 0;
    //For the i-th row of the subtasks grid the matrix  A blocks are shifted for (i-1) positions to the left,
    int shift = coords[0] + coords[2] * nproc_ij / c;
    if(shift > 0) {
        dst = (coords[1] +  c*nproc_ij - shift) % nproc_ij;
        src = (coords[1] + shift) % nproc_ij;

        MPI_Sendrecv_replace(A, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, rowcomm, &status);
    }

    shift = coords[1] + coords[2] * nproc_ij / c;
    if(shift > 0) {
        dst = (coords[0] +  c*nproc_ij - shift) % nproc_ij;
        src = (coords[0] + shift) % nproc_ij;

        MPI_Sendrecv_replace(BT, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, colcomm, &status);
    }

    mpi_check(MPI_Barrier(cartcomm));
    //finish initial shift


    int j_dst = (coords[1] +nproc_ij - 1) % nproc_ij; //used for BT's shift
    int j_src = (coords[1] + 1) % nproc_ij; //used for BT's shift
    int i_dst = (coords[0] +nproc_ij - 1) % nproc_ij; //used for BT's shift
    int i_src = (coords[0] + 1) % nproc_ij; //used for BT's shift

    //reference: http://www.hpcc.unn.ru/mskurs/ENG/PPT/pp08.pdf

    int stage = 0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            n_local, n_local, n_local,
            1, A, n_local, BT, n_local, 0, C, n_local);

    for (stage++; stage < nproc_ij/c; stage++) 
    {
        /* Send submatrix of A left and receive a new from right */
        MPI_Sendrecv_replace(A, n_local*n_local, MPI_DOUBLE, j_dst,
        datatag, j_src, datatag, rowcomm, &status);

        /* Send submatrix of B up and receive a new from below */
        MPI_Sendrecv_replace(BT, n_local*n_local, MPI_DOUBLE, i_dst,
        datatag, i_src, datatag, colcomm, &status);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_local, n_local, n_local,
                1, A, n_local, BT, n_local, 1, C, n_local);
    }

    //now do depth reduction
    int plane;
    MPI_Comm_rank(depthcomm, &plane);
    if (plane == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, C, n_local*n_local, MPI_DOUBLE, MPI_SUM, 0, depthcomm);
    } else {
        MPI_Reduce(C, C0, n_local*n_local, MPI_DOUBLE, MPI_SUM, 0, depthcomm);
    }

    //End timing
    t1 = MPI_Wtime() - t0;
    double avg_t;
    
    //end timing point
    //use reduction to collect the final timy_rank
    if (coords[2] == 0) 
    {
        MPI_Reduce(&t1, &avg_t, 1, MPI_DOUBLE, MPI_SUM, root, dim_ij_comm);
        avg_t /= (double) (nproc_ij*nproc_ij);
        
        if (my_rank == root) printf("[mmm25D]nproc=%d, N=%d, C=%d, Time=%.9f\n", nproc, n, c, avg_t);
    }
    
    #ifdef VERIFY
    gather_result(
        root, my_rank, coords, n, nproc_ij, n_local,
        dim_ij_comm, sendcounts, displs, subarrtype,
        C, M, NT, P
    );
    #endif
    
    MPI_Type_free(&subarrtype);
    MPI_Comm_free(&dim_ij_comm);
    mkl_free(A); mkl_free(BT); mkl_free(C); mkl_free(C0);
    MPI_Finalize();
    return 0;
}