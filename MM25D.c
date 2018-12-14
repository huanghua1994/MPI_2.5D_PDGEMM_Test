#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "mkl.h"

// #define VERIFY

#include "utils.c"

int main(int argc, char **argv) 
{
    int root = 0;
    MPI_Status status; 

    MPI_Init(&argc, &argv);
    
    int my_rank, nproc, c, nproc_ij;
    get_nproc_ij_c(argc, argv, &nproc, &my_rank, &c, &nproc_ij);

    int n = get_problem_size(argc, argv, nproc_ij, my_rank);
    int n_local = n / nproc_ij;
    int local_bs = n_local * n_local;
    int ntest;
    if (argc >= 4) ntest = atoi(argv[3]);
    if (ntest < 1 || ntest > 20) ntest = 10;
    if (my_rank == 0) 
    {
        printf("Test settings:\n");
        printf("  * Process grid : %d * %d * %d (c = %d)\n", nproc_ij, nproc_ij, c, c);
        printf("  * Matrix size  : Global N = %d, local N = %d\n", n, n_local);
        printf("  * Repeat tests : %d\n", ntest);
        printf("\n");
    }

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
    
    double *M, *N, *P;
    if (0 == my_rank) initial_matrix(&M, &N, &P, n, n, n);

    double *A  = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *B  = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *C  = (double *) mkl_malloc(local_bs * sizeof(double), 16);
    double *C0 = (double *) mkl_malloc(local_bs * sizeof(double), 16);

    // Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int *sendcounts = (int *) malloc(nproc_ij * nproc_ij * sizeof(int));
    int *displs = (int *) malloc(nproc_ij * nproc_ij * sizeof(int));
    init_subarrtype(root, my_rank, n, nproc_ij, n_local, &subarrtype, sendcounts, displs);

    // Distribute A & B on plane 0
    scatter_data(
        root, my_rank, my_plane, n, nproc_ij, local_bs,
        plane_comm, sendcounts, displs, subarrtype,
        A, B, M, N
    );
    MPI_Barrier(cart_comm);
    
    double st0, et0, st1, et1;
    double comm_t = 0.0, dgemm_t = 0.0, total_t = 0.0;
    
    for (int itest = 0; itest < ntest; itest++)
    {
        memset(C0, 0, local_bs * sizeof(double));
        memset(C , 0, local_bs * sizeof(double));
        MPI_Barrier(MPI_COMM_WORLD);
        
        st0 = MPI_Wtime();
        st1 = MPI_Wtime();
        
        // 1. Replicate input matrix on each plane
        MPI_Bcast(A, local_bs, MPI_DOUBLE, 0, dup_comm);
        MPI_Bcast(B, local_bs, MPI_DOUBLE, 0, dup_comm);
        
        // 2. Initial circular shift on A and B
        int dst, src, shift, datatag;
        // The shift formula here is different from the 2.5D paper, but it works...
        shift = i + k * nproc_ij / c;
        if (shift > 0) 
        {
            datatag = 0;
            dst = (j + c * nproc_ij - shift) % nproc_ij;
            src = (j + shift) % nproc_ij;
            MPI_Sendrecv_replace(
                A, local_bs, MPI_DOUBLE, dst, datatag, 
                src, datatag, row_comm, &status
            );
        }
        shift = j + k * nproc_ij / c;
        if (shift > 0) 
        {
            datatag = 1;
            dst = (i + c * nproc_ij - shift) % nproc_ij;
            src = (i + shift) % nproc_ij;
            MPI_Sendrecv_replace(
                B, local_bs, MPI_DOUBLE, dst, datatag, 
                src, datatag, col_comm, &status
            );
        }
        
        et1 = MPI_Wtime();
        comm_t += et1 - st1;
        
        // 3. Initial local DGEMM
        st1 = MPI_Wtime();
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n_local, n_local, n_local,
            1, A, n_local, B, n_local, 0, C0, n_local
        );
        et1 = MPI_Wtime();
        dgemm_t += et1 - st1;

        // 4. Do nproc_ij / c steps of Cannon's algorithm
        int j_dst = (j + 1) % nproc_ij;
        int i_dst = (i + 1) % nproc_ij; 
        int j_src = (j - 1 + nproc_ij) % nproc_ij;
        int i_src = (i - 1 + nproc_ij) % nproc_ij; 
        for (int stage = 1; stage < nproc_ij / c; stage++) 
        {
            datatag = stage + 1;
            
            st1 = MPI_Wtime();
            
            // (1) Send A block to the right and receive a new from the left
            MPI_Sendrecv_replace(
                A, local_bs, MPI_DOUBLE, j_dst, datatag, 
                j_src, datatag, row_comm, &status
            );

            // (2) Send B block to the below and receive a new from the up
            MPI_Sendrecv_replace(
                B, local_bs, MPI_DOUBLE, i_dst, datatag, 
                i_src, datatag, col_comm, &status
            );
            
            et1 = MPI_Wtime();
            comm_t += et1 - st1;
            
            // (3) Local DGEMM
            st1 = MPI_Wtime();
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_local, n_local, n_local,
                1, A, n_local, B, n_local, 1, C0, n_local
            );
            et1 = MPI_Wtime();
            dgemm_t += et1 - st1;
        }

        // 5. Reduce sum the result to processes on plane 0
        st1 = MPI_Wtime();
        MPI_Reduce(C0, C, local_bs, MPI_DOUBLE, MPI_SUM, 0, dup_comm);
        et1 = MPI_Wtime();
        dgemm_t += et1 - st1;
        
        et0 = MPI_Wtime();
        total_t += et0 - st0;
    }
    
    double GFlop = 2.0 * (double) n * (double) n * (double) n * (double) ntest / 1000000000.0;
    
    double avg_comm_t, avg_dgemm_t, avg_total_t;
    
    if (my_plane == 0) 
    {
        MPI_Reduce(&comm_t,  &avg_comm_t,  1, MPI_DOUBLE, MPI_SUM, root, plane_comm);
        MPI_Reduce(&dgemm_t, &avg_dgemm_t, 1, MPI_DOUBLE, MPI_SUM, root, plane_comm);
        MPI_Reduce(&total_t, &avg_total_t, 1, MPI_DOUBLE, MPI_SUM, root, plane_comm);
        avg_comm_t  /= (double) (nproc_ij * nproc_ij);
        avg_dgemm_t /= (double) (nproc_ij * nproc_ij);
        avg_total_t /= (double) (nproc_ij * nproc_ij);
        
        if (my_rank == root) 
        {
            printf("PDGEMM 2.5D algorithm %d runs average timing:\n", ntest);
            printf("  * Communication = %.2lf (s)\n", avg_comm_t);
            printf("  * Local DGEMM   = %.2lf (s), %.2lf GFlops\n", avg_dgemm_t, GFlop / avg_dgemm_t);
            printf("  * Overall       = %.2lf (s), %.2lf GFlops\n", avg_total_t, GFlop / avg_total_t);
        }
    }
    
    #ifdef VERIFY
    gather_result(
        root, my_rank, my_plane, n, nproc_ij, local_bs,
        plane_comm, sendcounts, displs, subarrtype,
        C, M, N, P
    );
    #endif
    
    free(sendcounts);
    free(displs);
    mkl_free(A);
    mkl_free(B);
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