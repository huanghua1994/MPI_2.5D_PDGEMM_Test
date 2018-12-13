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
    MPI_Comm cart_comm;
    int dims[3] = {nproc_ij, nproc_ij, c};
    int periods[3] = {1, 1, 0};
    int coords[3];
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 3, coords);

    // Split the cart_comm into row_comm, col_comm, dup_comm, and plane_comm
    // row_comm: P(i, *, k), col_comm: P(*, j, k), dup_comm: P(i, j, *), plane_comm: P(*, *, k)
    MPI_Comm row_comm, col_comm, dup_comm, plane_comm;
    int remain_i[3]  = {1, 0, 0};
    int remain_j[3]  = {0, 1, 0};
    int remain_k[3]  = {0, 0, 1};
    int remain_ij[3] = {1, 1, 0};
    MPI_Cart_sub(cart_comm, remain_j,  &row_comm);
    MPI_Cart_sub(cart_comm, remain_i,  &col_comm);
    MPI_Cart_sub(cart_comm, remain_k,  &dup_comm);
    MPI_Cart_sub(cart_comm, remain_ij, &plane_comm);

    double *M, *NT, *P;
    if (0 == my_rank) initial_matrix(&M, &NT, &P, n, n, n);

    int i, j, k;
    double *A  = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *BT = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *C  = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *C0 = (double *) mkl_malloc(local_bs * sizeof(double), 16);

    //Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int sendcounts[nproc_ij*nproc_ij];
    int displs[nproc_ij*nproc_ij]; //value in block (resized) count
    init_subarrtype(root, my_rank, n, nproc_ij, n_local, &subarrtype, sendcounts, displs);

    //Now A/BT local are ready accorss layer 0.
    scatter_data(root, my_rank, coords, n, nproc_ij, n_local,
            plane_comm, sendcounts, displs, subarrtype,
            A, BT, M, NT);

    mpi_check(MPI_Barrier(cart_comm));
    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();
    //the main MMM25D algorithm part
    //The real start point.
    mpi_check(MPI_Bcast(A, n_local*n_local, MPI_DOUBLE, 0, dup_comm));
    mpi_check(MPI_Bcast(BT,n_local*n_local, MPI_DOUBLE, 0, dup_comm));

    //first need shift both A and B.
    int dst, src; //use for shift
    int datatag = 0;
    //For the i-th row of the subtasks grid the matrix  A blocks are shifted for (i-1) positions to the left,
    int shift = coords[0] + coords[2] * nproc_ij / c;
    if(shift > 0) {
        dst = (coords[1] +  c*nproc_ij - shift) % nproc_ij;
        src = (coords[1] + shift) % nproc_ij;

        MPI_Sendrecv_replace(A, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, row_comm, &status);
    }

    shift = coords[1] + coords[2] * nproc_ij / c;
    if(shift > 0) {
        dst = (coords[0] +  c*nproc_ij - shift) % nproc_ij;
        src = (coords[0] + shift) % nproc_ij;

        MPI_Sendrecv_replace(BT, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, col_comm, &status);
    }

    mpi_check(MPI_Barrier(cart_comm));
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
        datatag, j_src, datatag, row_comm, &status);

        /* Send submatrix of B up and receive a new from below */
        MPI_Sendrecv_replace(BT, n_local*n_local, MPI_DOUBLE, i_dst,
        datatag, i_src, datatag, col_comm, &status);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_local, n_local, n_local,
                1, A, n_local, BT, n_local, 1, C, n_local);
    }

    //now do depth reduction
    int plane;
    MPI_Comm_rank(dup_comm, &plane);
    if (plane == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, C, n_local*n_local, MPI_DOUBLE, MPI_SUM, 0, dup_comm);
    } else {
        MPI_Reduce(C, C0, n_local*n_local, MPI_DOUBLE, MPI_SUM, 0, dup_comm);
    }

    //End timing
    t1 = MPI_Wtime() - t0;
    double avg_t;
    
    //end timing point
    //use reduction to collect the final timy_rank
    if (coords[2] == 0) 
    {
        MPI_Reduce(&t1, &avg_t, 1, MPI_DOUBLE, MPI_SUM, root, plane_comm);
        avg_t /= (double) (nproc_ij*nproc_ij);
        
        if (my_rank == root) printf("[mmm25D]nproc=%d, N=%d, C=%d, Time=%.9f\n", nproc, n, c, avg_t);
    }
    
    #ifdef VERIFY
    gather_result(
        root, my_rank, coords, n, nproc_ij, n_local,
        plane_comm, sendcounts, displs, subarrtype,
        C, M, NT, P
    );
    #endif
    
    
    mkl_free(A); 
    mkl_free(BT); 
    mkl_free(C); 
    mkl_free(C0);
    
    MPI_Type_free(&subarrtype);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&dup_comm);
    MPI_Comm_free(&plane_comm);
    MPI_Finalize();
    
    return 0;
}