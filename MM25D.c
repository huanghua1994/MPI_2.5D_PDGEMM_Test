#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include "mpi.h"
#include "mkl.h"

//#define VERIFY

#include "utils.c"

#define N_DUP    4
MPI_Comm    dup_comms[N_DUP];
MPI_Status  status[N_DUP];
MPI_Request reqs[N_DUP], reqs0[N_DUP];
int spos[N_DUP + 1], blklen[N_DUP];
int row_spos[N_DUP + 1], row_blklen[N_DUP];

void duplicate_comms(MPI_Comm dup_comm, int nrow, int ncol)
{    
    int remainder  = nrow % N_DUP;
    int block_size = nrow / N_DUP;
    for (int i = 0; i < remainder; i++)
    {
        row_blklen[i] = block_size + 1;
        blklen[i] = row_blklen[i] * ncol;
    }
    for (int i = remainder; i < N_DUP; i++)
    {
        row_blklen[i] = block_size;
        blklen[i] = row_blklen[i] * ncol;
    }
    
    row_spos[0] = spos[0] = 0;
    for (int i = 0; i < N_DUP; i++)
    {
        MPI_Comm_dup(dup_comm, &dup_comms[i]);
        spos[i + 1] = spos[i] + blklen[i];
        row_spos[i + 1] = row_spos[i] + row_blklen[i];
    }
}

void MM25D_Cannon_steps(
    int i, int j, int k, int nproc_ij, int c, 
    int n_local, int local_bs,
    double *A, double *B, double *A0, double *B0, double *C,
    MPI_Comm row_comm, MPI_Comm col_comm, 
    double *p2p_t, double *dgemm_t
)
{
    MPI_Status status;
    double st1, et1;
    
    double *sendA, *recvA, *sendB, *recvB, *tmpptr;
    MPI_Status sta0, sta1;
    MPI_Request req0, req1, req2;
    
    sendA = A;  recvA = A0;
    sendB = B;  recvB = B0;
    
    // (1) Initial circular shift on A and B
    st1 = MPI_Wtime();
    int shift = k * (nproc_ij / c);
    int dstA = (j - i + shift + c * nproc_ij) % nproc_ij;
    int dstB = (i - j + shift + c * nproc_ij) % nproc_ij;
    MPI_Isend(sendA, local_bs, MPI_DOUBLE, dstA, 0, row_comm, &req2);
    MPI_Isend(sendB, local_bs, MPI_DOUBLE, dstB, 1, col_comm, &req2);
    MPI_Irecv(recvA, local_bs, MPI_DOUBLE, MPI_ANY_SOURCE, 0, row_comm, &req0);
    MPI_Irecv(recvB, local_bs, MPI_DOUBLE, MPI_ANY_SOURCE, 1, col_comm, &req1);
    MPI_Wait(&req0, &sta0);
    MPI_Wait(&req1, &sta1); 
    
    et1 = MPI_Wtime();
    p2p_t[0] += et1 - st1;
    
    // (2) Initial local DGEMM
    st1 = MPI_Wtime();
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n_local, n_local, n_local,
        1, recvA, n_local, recvB, n_local, 0, C, n_local
    );
    et1 = MPI_Wtime();
    dgemm_t[0] += et1 - st1;

    tmpptr = sendA; sendA = recvA; recvA = tmpptr;
    tmpptr = sendB; sendB = recvB; recvB = tmpptr;
    
    // (3) Do nproc_ij / c steps of Cannon's algorithm
    int j_dst = (j + 1) % nproc_ij;
    int i_dst = (i + 1) % nproc_ij; 
    int j_src = (j - 1 + nproc_ij) % nproc_ij;
    int i_src = (i - 1 + nproc_ij) % nproc_ij; 
    for (int stage = 1; stage < nproc_ij / c; stage++) 
    {
        int datatag = stage + 1;
        
        st1 = MPI_Wtime();
        
        // Send A block to the right and receive a new from the left
        MPI_Isend(sendA, local_bs, MPI_DOUBLE, j_dst, datatag, row_comm, &req2);
        MPI_Irecv(recvA, local_bs, MPI_DOUBLE, j_src, datatag, row_comm, &req0);

        // Send B block to the below and receive a new from the up
        MPI_Isend(sendB, local_bs, MPI_DOUBLE, i_dst, datatag, col_comm, &req2);
        MPI_Irecv(recvB, local_bs, MPI_DOUBLE, i_src, datatag, col_comm, &req1);
        
        MPI_Wait(&req0, &sta0);
        MPI_Wait(&req1, &sta1);
        
        et1 = MPI_Wtime();
        p2p_t[0] += et1 - st1;
        
        // Local DGEMM
        st1 = MPI_Wtime();
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n_local, n_local, n_local,
            1, recvA, n_local, recvB, n_local, 1, C, n_local
        );
        et1 = MPI_Wtime();
        dgemm_t[0] += et1 - st1;
        
        tmpptr = sendA; sendA = recvA; recvA = tmpptr;
        tmpptr = sendB; sendB = recvB; recvB = tmpptr;
    }
}

int main(int argc, char **argv) 
{
    int root = 0;
    
    // Disable memory mapped malloc, previously done in MA_init() 
    // for caching page registrations 
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);

    MPI_Init(&argc, &argv);
    
    int my_rank, nproc, c, nproc_ij;
    get_nproc_ij_c(argc, argv, &nproc, &my_rank, &c, &nproc_ij);

    int n = get_problem_size(argc, argv, nproc_ij, my_rank);
    int n_local = n / nproc_ij;
    int local_bs = n_local * n_local;
    int ntest = 10;
    if (argc >= 4) ntest = atoi(argv[3]);
    if (ntest < 1 || ntest > 50) ntest = 10;
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
    
    duplicate_comms(dup_comm, n_local, n_local);

    int plane;
    MPI_Comm_rank(dup_comm, &plane);
    
    double *M, *N, *P, *Q;
    if (0 == my_rank) initial_matrix(&M, &N, &P, &Q, n);

    double *matbuf = (double *) mkl_malloc(8 * local_bs * sizeof(double), 16);
    double *A  = matbuf + local_bs * 0;
    double *B  = matbuf + local_bs * 1;
    double *C  = matbuf + local_bs * 2;
    double *D  = matbuf + local_bs * 3;
    double *A0 = matbuf + local_bs * 4;  // A0 & B0 is used as a buffer for sendrecv
    double *B0 = matbuf + local_bs * 5; 
    double *C0 = matbuf + local_bs * 6;
    double *D0 = matbuf + local_bs * 7;

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
    double p2p_t = 0.0, bcast_t = 0.0, reduce_t = 0.0, allreduce_t = 0.0, dgemm_t = 0.0, total_t = 0.0;
    
    // Backup A & B since they will be changed after computing C := A * B
    //memcpy(A0, A, sizeof(double) * local_bs);
    //memcpy(B0, B, sizeof(double) * local_bs);
    
    for (int itest = 0; itest < ntest; itest++)
    {
        // Recover the original A & B
        //memcpy(A, A0, sizeof(double) * local_bs);
        //memcpy(B, B0, sizeof(double) * local_bs);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        st0 = MPI_Wtime();
        
        // 1.1 Replicate A on each plane
        st1 = MPI_Wtime();
        for (int ii = 0; ii < N_DUP; ii++)
            MPI_Ibcast(A + spos[ii], blklen[ii], MPI_DOUBLE, 0, dup_comms[ii], &reqs[ii]);
        MPI_Waitall(N_DUP, &reqs[0], &status[0]);
        //MPI_Bcast(A, local_bs, MPI_DOUBLE, 0, dup_comm);
        et1 = MPI_Wtime();
        bcast_t += et1 - st1;
        
        // 1.2 Copy B := A; backup the broadcast A to save a broadcast in D := A * C
        memcpy(B, A, sizeof(double) * local_bs);
        memcpy(D0, A, sizeof(double) * local_bs);

        // 2. Do nproc_ij/c steps of Canon's algorithm to get C := A * B
        MM25D_Cannon_steps(
            i, j, k, nproc_ij, c, n_local, local_bs, A, B, A0, B0, 
            C0, row_comm, col_comm, &p2p_t, &dgemm_t
        );
       
        // 3. Reduce sum C to plane 0
        st1 = MPI_Wtime();
        for (int ii = 0; ii < N_DUP; ii++)
            MPI_Ireduce(C0 + spos[ii], C + spos[ii], blklen[ii], MPI_DOUBLE, MPI_SUM, 0, dup_comms[ii], &reqs[ii]);
        MPI_Waitall(N_DUP, &reqs[0], &status[0]);
        //MPI_Reduce(C0, C, local_bs, MPI_DOUBLE, MPI_SUM, 0, dup_comm);
        et1 = MPI_Wtime();
        reduce_t += et1 - st1;
        
        // 4.1 Replicate C on each plane
        st1 = MPI_Wtime();
        for (int ii = 0; ii < N_DUP; ii++)
            MPI_Ibcast(C + spos[ii], blklen[ii], MPI_DOUBLE, 0, dup_comms[ii], &reqs[ii]);
        MPI_Waitall(N_DUP, &reqs[0], &status[0]);
        //MPI_Bcast(C, local_bs, MPI_DOUBLE, 0, dup_comm);
        et1 = MPI_Wtime();
        bcast_t += et1 - st1;
        
        // 4.2 Backup C and Recover A
        memcpy(C0, C, sizeof(double) * local_bs);
        memcpy(A, D0, sizeof(double) * local_bs);
        
        // 5. Do nproc_ij/c steps of Canon's algorithm
        MM25D_Cannon_steps(
            i, j, k, nproc_ij, c, n_local, local_bs, A, C0, A0, B0, 
            D0, row_comm, col_comm, &p2p_t, &dgemm_t
        );
       
        // 6. Reduce sum C to processes on plane 0
        st1 = MPI_Wtime();
        for (int ii = 0; ii < N_DUP; ii++)
            MPI_Ireduce(D0 + spos[ii], D + spos[ii], blklen[ii], MPI_DOUBLE, MPI_SUM, 0, dup_comms[ii], &reqs[ii]);
        MPI_Waitall(N_DUP, &reqs[0], &status[0]);
        //MPI_Reduce(D0, D, local_bs, MPI_DOUBLE, MPI_SUM, 0, dup_comm);
        et1 = MPI_Wtime();
        reduce_t += et1 - st1;
        
        et0 = MPI_Wtime();
        total_t += et0 - st0;
        
        #ifdef VERIFY
        if (itest == 0)
        {
            gather_result(
                root, my_rank, my_plane, n, nproc_ij, local_bs,
                plane_comm, sendcounts, displs, subarrtype,
                C, D, M, N, P, Q
            );
        }
        #endif
    }
    
    double GFlop = 4.0 * (double) n * (double) n * (double) n * (double) ntest / 1000000000.0;
    
    if (my_plane == 0) 
    {
        double max_p2p_t, max_bcast_t, max_reduce_t, max_dgemm_t, max_total_t, max_comm_t, nproj_ij2;
        MPI_Reduce(&p2p_t,       &max_p2p_t,       1, MPI_DOUBLE, MPI_MAX, root, plane_comm);
        MPI_Reduce(&bcast_t,     &max_bcast_t,     1, MPI_DOUBLE, MPI_MAX, root, plane_comm);
        MPI_Reduce(&reduce_t,    &max_reduce_t,    1, MPI_DOUBLE, MPI_MAX, root, plane_comm);
        MPI_Reduce(&dgemm_t,     &max_dgemm_t,     1, MPI_DOUBLE, MPI_MAX, root, plane_comm);
        MPI_Reduce(&total_t,     &max_total_t,     1, MPI_DOUBLE, MPI_MAX, root, plane_comm);
        nproj_ij2 = (double) (nproc_ij * nproc_ij);
        max_comm_t = max_p2p_t + max_bcast_t + max_reduce_t;
        
        if (my_rank == root) 
        {
            printf("PDGEMM 2.5D algorithm %d runs max timing:\n", ntest);
            printf("  Communication time:\n");
            printf("    * P2P        : %.2lf (s)\n", max_p2p_t);
            printf("    * Bcast      : %.2lf (s)\n", max_bcast_t);
            printf("    * Reduce     : %.2lf (s)\n", max_reduce_t);
            printf("    * Total      : %.2lf (s)\n", max_comm_t);
            printf("  Local DGEMM = %.2lf (s), %.2lf GFlops\n", max_dgemm_t, GFlop / max_dgemm_t);
            printf("  Overall = %.2lf (s), %.2lf GFlops\n", max_total_t, GFlop / max_total_t);
        }
    }
    
    free(sendcounts);
    free(displs);
    mkl_free(matbuf);
    
    MPI_Type_free(&subarrtype);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&dup_comm);
    MPI_Comm_free(&plane_comm);
    MPI_Finalize();
    
    return 0;
}