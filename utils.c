
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

int get_problem_size(int argc, char **argv, int nproc_ij, int my_rank) 
{
    int n = nproc_ij; 
    if(argc > 1) 
    {
        n = atoi(argv[1]);
        // Pad the matrix to so that each process get the same size
        int n_local = (n - 1) / nproc_ij + 1;
        if (n_local * nproc_ij != n) 
        {
            if (my_rank == 0) printf("Warning: Padding the problem size from %d to %d\n", n, n_local * nproc_ij);
            n = n_local * nproc_ij;
        }
    }
    return n;
}

// An elegant by abstracted way: https://stackoverflow.com/questions/9269399/sending-blocks-of-2d-array-in-c-using-mpi/9271753#9271753
void init_subarrtype(
    int root, int my_rank, int n, int nproc_ij, int n_local,
    MPI_Datatype* subarrtype_addr, int *sendcounts, int *displs
) 
{
    int sizes[2]    = {n, n};         
    int subsizes[2] = {n_local, n_local};
    int starts[2]   = {0, 0};

    MPI_Datatype type;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, n_local * sizeof(double), subarrtype_addr);
    MPI_Type_commit(subarrtype_addr);
    if (my_rank == root) 
    {
        for (int i = 0; i < nproc_ij * nproc_ij; i++) sendcounts[i] = 1;
        int disp = 0;
        for (int i = 0; i < nproc_ij; i++) 
        {
            for (int j = 0; j < nproc_ij; j++) 
            {
                displs[i * nproc_ij + j] = disp;
                disp += 1;
            }
            disp += (n_local - 1) * nproc_ij;
        }
    }
}

void initial_matrix(double **A_addr, double **B_addr, double **C_addr, int m, int n, int k) 
{
    double* A = (double *) mkl_malloc(m * k * sizeof(double), 16);
    double* B = (double *) mkl_malloc(n * k * sizeof(double), 16);
    double* C = (double *) mkl_malloc(m * n * sizeof(double), 16);
    *A_addr = A;
    *B_addr = B;
    *C_addr = C;
    
    srand(time(NULL));
    double RMAX = (double) RAND_MAX;
    for (int i = 0; i < m * k; i++) A[i] = (double) rand() / RMAX;
    for (int i = 0; i < k * n; i++) B[i] = (double) rand() / RMAX;
}

int check_result(double *A, double *B, double *C, int m, int n, int k) 
{
    int err_counts = 0;
    double *C_ref = (double *) mkl_malloc(m * n * sizeof(double), 16);
    
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, 1.0, A, k, B, k, 0.0, C_ref, n
    );

    for (int i = 0; i < m * n; i++) 
        err_counts += (fabs(C[i] - C_ref[i]) < 0.0001 ? 0 : 1);

    mkl_free(C_ref);
    
    return err_counts;
}

void scatter_data(
    int root, int my_rank, int my_plane, int n, int nproc_ij, int local_bs,
    MPI_Comm plane_comm, int *sendcounts, int *displs, MPI_Datatype subarrtype,
    double *A, double *B, double *M, double *N
) 
{

    if (my_plane == 0) 
    {
        MPI_Scatterv(
            M, sendcounts, displs, subarrtype,
            A, local_bs, MPI_DOUBLE, root, plane_comm
        );

        MPI_Scatterv(
            N, sendcounts, displs, subarrtype,
            B, local_bs, MPI_DOUBLE, root, plane_comm
        );
    }
}

void gather_result(
    int root, int my_rank, int my_plane, int n, int nproc_ij, int local_bs,
    MPI_Comm plane_comm, int *sendcounts,  int *displs, MPI_Datatype subarrtype,
    double *C, double *M, double *N, double *P
) 
{
    if (my_plane == 0) 
    {
        MPI_Gatherv(
            C, local_bs,  MPI_DOUBLE, P, 
            sendcounts, displs, subarrtype, root, plane_comm
        );
    }

    if (my_rank == root) 
    {
        int err_counts = check_result(M, N, P, n, n, n); 
        if (err_counts) 
        {
            fprintf(stderr, "Check failed: %d errors\n", err_counts);
        } else {
            printf("Result is correct\n");
        }
        mkl_free(M);
        mkl_free(N);
        mkl_free(P);
    }
}

