#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "mkl.h"

#define VERIFY

#include "utils.c"

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
    if (my_rank == 0) printf("nproc = %d, n = %d, nproc_ij = %d, c = %d\n", nproc, n, nproc_ij, c);

    //prepare for the cartesian topology
    MPI_Comm cart_comm;
    int dims[3] = {nproc_ij, nproc_ij, c};
    int periods[3] = {1, 1, 0};
    int coords[3];
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 3, coords);
    
    int i, j, k, my_row, my_col, my_plane;
    i = my_row   = coords[0]; 
    j = my_col   = coords[1]; 
    k = my_plane = coords[2]; 

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

    int plane;
    MPI_Comm_rank(dup_comm, &plane);
    
    double *M, *NT, *P;
    if (0 == my_rank) initial_matrix(&M, &NT, &P, n, n, n);

    double *A  = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *BT = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *C  = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *C0 = (double *) mkl_malloc(local_bs * sizeof(double), 16);

    //Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int *sendcounts = (int *) malloc(nproc_ij * nproc_ij * sizeof(int));
    int *displs = (int *) malloc(nproc_ij * nproc_ij * sizeof(int));
    init_subarrtype(root, my_rank, n, nproc_ij, n_local, &subarrtype, sendcounts, displs);

    // Distribute A & BT on plane 0
    scatter_data(
        root, my_rank, my_plane, n, nproc_ij, n_local,
        plane_comm, sendcounts, displs, subarrtype,
        A, BT, M, NT
    );
    MPI_Barrier(cart_comm);
    
    double t0, t1;
    
    t0 = MPI_Wtime();
    
    // 1. Replicate input matrix on each plane
    MPI_Bcast(A,  local_bs, MPI_DOUBLE, 0, dup_comm);
    MPI_Bcast(BT, local_bs, MPI_DOUBLE, 0, dup_comm);

    // 2. Initial circular shift on A and B
    int dst, src, shift, datatag = 0;
    shift = i + k * nproc_ij / c;
    if (shift > 0) 
    {
        dst = (j + c * nproc_ij - shift) % nproc_ij;
        src = (j + shift) % nproc_ij;

        MPI_Sendrecv_replace(A, local_bs, MPI_DOUBLE, dst,
        datatag, src, datatag, row_comm, &status);
    }
    shift = j + k * nproc_ij / c;
    if (shift > 0) 
    {
        dst = (i + c * nproc_ij - shift) % nproc_ij;
        src = (i + shift) % nproc_ij;

        MPI_Sendrecv_replace(BT, local_bs, MPI_DOUBLE, dst,
        datatag, src, datatag, col_comm, &status);
    }
    MPI_Barrier(cart_comm);

    int j_dst = (j + nproc_ij - 1) % nproc_ij;
    int j_src = (j + 1) % nproc_ij;
    int i_dst = (i + nproc_ij - 1) % nproc_ij; 
    int i_src = (i + 1) % nproc_ij; 

    // 3. Initial local DGEMM
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        n_local, n_local, n_local,
        1, A, n_local, BT, n_local, 0, C, n_local
    );

    // 4. Do nproc_ij / c steps of Cannon's algorithm
    for (int stage = 1; stage < nproc_ij / c; stage++) 
    {
        // (1) Send A block to the left and receive a new from the right
        MPI_Sendrecv_replace(
            A, local_bs, MPI_DOUBLE, j_dst,
            datatag, j_src, datatag, row_comm, &status
        );

        // (2) Send B block to the up and receive a new from the below
        MPI_Sendrecv_replace(
            BT, local_bs, MPI_DOUBLE, i_dst,
            datatag, i_src, datatag, col_comm, &status
        );
        
        // (3) Local DGEMM
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            n_local, n_local, n_local,
            1, A, n_local, BT, n_local, 1, C, n_local
        );
    }

    // 5. Reduce sum the result to processes on plane 0
    if (plane == 0) MPI_Reduce(MPI_IN_PLACE, C, local_bs, MPI_DOUBLE, MPI_SUM, 0, dup_comm);
    else            MPI_Reduce(C, C0, local_bs, MPI_DOUBLE, MPI_SUM, 0, dup_comm);

    t1 = MPI_Wtime() - t0;
    double avg_t;
    

    if (k == 0) 
    {
        MPI_Reduce(&t1, &avg_t, 1, MPI_DOUBLE, MPI_SUM, root, plane_comm);
        avg_t /= (double) (nproc_ij*nproc_ij);
        
        if (my_rank == root) printf("PDGEMM 2.5D time = %.4lf (s) \n", avg_t);
    }
    
    #ifdef VERIFY
    gather_result(
        root, my_rank, my_plane, n, nproc_ij, n_local,
        plane_comm, sendcounts, displs, subarrtype,
        C, M, NT, P
    );
    #endif
    
    free(sendcounts);
    free(displs);
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