
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

int get_problem_size(int argc, char **argv, int dim_sz, int my_rank) 
{
    int n = dim_sz; 
    if(argc > 1) 
    {
        n = atoi(argv[1]);
        // Pad the matrix to so that each process get the same size
        int local_n = (n - 1) / dim_sz + 1;
        if (local_n * dim_sz != n) 
        {
            if (my_rank == 0) printf("Warning: Padding the problem size from %d to %d, grid size is %d \n", n, local_n * dim_sz, dim_sz);
            n = local_n * dim_sz;
        } else {
          if (my_rank == 0) printf("Problem size is %d, grid size is %d \n", n, dim_sz);
        }
    }
    return n;
}

void init_subarrtype(
    int root, int my_rank, int n, int dim_sz, int local_n,
    MPI_Datatype* subarrtype_addr, int *sendcounts, int *displs
) 
{
    int sizes[2]    = {n, n};         
    int subsizes[2] = {local_n, local_n};
    int starts[2]   = {0, 0};

    MPI_Datatype type;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, local_n * sizeof(double), subarrtype_addr);
    MPI_Type_commit(subarrtype_addr);
    if (my_rank == root) 
    {
        for (int i = 0; i < dim_sz * dim_sz; i++) sendcounts[i] = 1;
        int disp = 0;
        for (int i = 0; i < dim_sz; i++) 
        {
            for (int j = 0; j < dim_sz; j++) 
            {
                displs[i * dim_sz + j] = disp;
                disp += 1;
            }
            disp += (local_n - 1) * dim_sz;
        }
    }
}

void initial_matrix(double** A_addr, double** BT_addr, double** C_addr, int m, int n, int k) 
{
    double* A  = (double *) mkl_malloc(m * k * sizeof(double), 16);
    double* BT = (double *) mkl_malloc(n * k * sizeof(double), 16);
    double* C  = (double *) mkl_malloc(m * n * sizeof(double), 16);
    *A_addr  = A;
    *BT_addr = BT;
    *C_addr  = C;
    
    srand(time(NULL));
    double RMAX = (double) RAND_MAX;
    for (int i = 0; i < m * k; i++) A[i] = (double) rand() / RMAX;
    for (int i = 0; i < k * n; i++) BT[i] = (double) rand() / RMAX;
}

int check_result(double *A, double *BT, double *C, int m, int n, int k, int C_transposed) 
{
    int err_counts = 0;
    double *C_ref = (double *) mkl_malloc(m * n * sizeof(double), 16);
    
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        m, n, k, 1.0, A, k, BT, k, 0.0, C_ref, n
    );

    if (C_transposed)
    {
        for (int i = 0; i < m; i++) 
            for (int j = 0; j < n; j++) 
                err_counts += (fabs(C[j * m + i] - C_ref[i * n + j]) < 0.0001 ? 0 : 1);
    } else {
        for(int i = 0; i < m * n; i++) 
            err_counts += (fabs(C[i] - C_ref[i]) < 0.0001 ? 0 : 1);
    }
    mkl_free(C_ref);
    
    return err_counts;
}

void scatter_data(
    int root, int my_rank, int my_plane, int n, int nproc_ij, int n_local,
    MPI_Comm plane_comm, int *sendcounts, int *displs, MPI_Datatype subarrtype,
    double *A, double *BT, double* M, double* NT
) 
{

    if (my_plane == 0) 
    {
        MPI_Scatterv(
            M, sendcounts,  displs, subarrtype,
            A, n_local*n_local, MPI_DOUBLE, root, plane_comm
        );
    }
    
    int displsBT[nproc_ij*nproc_ij];
    int i,j;
    if (root == my_rank) 
    {
        for (int i = 0; i < nproc_ij; i++) 
        {
            for (int j = 0; j < nproc_ij; j++) 
            {
                displsBT[i*nproc_ij+j] = displs[j*nproc_ij+i];
            }
        }
    }

    if (my_plane == 0) 
    {
        MPI_Scatterv(
            NT, sendcounts, displsBT, subarrtype,
            BT, n_local*n_local, MPI_DOUBLE, root, plane_comm
        );
    }
}

void gather_result(
    int root, int my_rank, int my_plane, int n, int nproc_ij, int n_local,
    MPI_Comm plane_comm, int *sendcounts,  int *displs, MPI_Datatype subarrtype,
    double *C, double *M, double *NT, double *P
) 
{
    if (my_plane == 0) 
    {
        MPI_Gatherv(
            C, n_local*n_local,  MPI_DOUBLE, P, 
            sendcounts, displs, subarrtype, root, plane_comm
        );
    }

    if (my_rank == root) 
    {
        int err_counts = check_result(M, NT, P, n, n, n, 0); 
        if (err_counts) 
        {
            fprintf(stderr, "Check failed: %d errors\n", err_counts);
        } else {
            printf("Result is correct\n");
        }
        mkl_free(M);
        mkl_free(NT);
        mkl_free(P);
    }
}

