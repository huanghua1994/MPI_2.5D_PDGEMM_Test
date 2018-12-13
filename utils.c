

#define mpi_check(stmt) do {                               \
        int err = stmt;                                    \
        if (err != MPI_SUCCESS) {                          \
            fprintf(stderr, "MPI_ERROR %d: from running stmt %s\n", err ,#stmt); \
            MPI_Finalize(); \
            exit(err);  \
        }                                                  \
    } while(0)

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
        // Padding the matrix to so that each process get the same size
        int per_n = (n - 1) / dim_sz + 1;
        if (per_n * dim_sz != n) 
        {
            if (my_rank == 0) printf("Warning: Padding the problem size from %d to %d, grid size is %d \n", n, per_n * dim_sz, dim_sz);
            n = per_n * dim_sz;
        } else {
          if (my_rank == 0) printf("Problem size is %d, grid size is %d \n", n, dim_sz);
        }
    }
    return n;
}

void init_subarrtype(int root, int me,
        int n, int dim_sz, int per_n,
        MPI_Datatype* subarrtype_addr, int sendcounts[], int displs[]) 
{
    int sizes[2]    = {n, n};         /* global size */
    int subsizes[2] = {per_n, per_n}; /* local size */
    int starts[2]   = {0,0};          /* where this one starts */

    MPI_Datatype type;
    mpi_check(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type));
    mpi_check(MPI_Type_create_resized(type, 0, per_n*sizeof(double), subarrtype_addr));
    mpi_check(MPI_Type_commit(subarrtype_addr));
    int i,j;
    if(me == root) {
        for (i=0; i< dim_sz*dim_sz; i++) { sendcounts[i] = 1; }
        int disp = 0;
        for (i=0; i<dim_sz; i++) {
            for (j=0; j<dim_sz; j++) {
                displs[i*dim_sz+j] = disp;
                disp += 1;
            }
            disp += (per_n-1)*dim_sz;
        }
    }
}


void initial_matrix(double** A_addr, double** BT_addr, double** C_addr, int m, int n, int c) 
{
    double* A = (double*)mkl_malloc(m * c * sizeof(double) , 16);
    double* BT = (double*)mkl_malloc(n * c * sizeof(double), 16);
    double* C = (double*)mkl_malloc(m * n * sizeof(double), 16);
    *A_addr = A;
    *BT_addr = BT;
    *C_addr = C;
    //use random number for input
    int i,j,k;

    srand((unsigned)time(NULL));
    for(i = 0; i < m; i++) {
        for(k = 0; k < c; k++) {
            A[i*c+k] = ((double)rand()/(double)RAND_MAX);
        }
    }
    for(j = 0; j < n; j++) {
        for(k = 0; k < c; k++) {
            BT[j*c+k] = ((double)rand()/(double)RAND_MAX);
        }
    }
}



int check_result(double* A, double* BT, double* C, int m, int n, int c, int transposed) 
{
    int err_c = 0; //how many errors found
    int i, j, k;
    double* C_ref = (double*)mkl_malloc(m * n * sizeof(double), 16); //with zeroed
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            m, n, c,
            1, A, c, BT, c, 0, C_ref, n);


    if(transposed){
        for(i = 0; i < m; i++) {
            for(j = 0; j < n; j++) {
                err_c += (fabs(C[j*m + i] - C_ref[i*n+j]) < 0.0001 ? 0 : 1);
            }
        }
    } else {
        //do compare
        for(i = 0; i < m * n ; i++) {
            err_c += (fabs(C[i] - C_ref[i]) < 0.0001 ? 0 : 1);
        }
    }
    mkl_free(C_ref);
    return err_c;
}

void scatter_data(int root, int me, int coords[3],
        int n, int p_gridsize, int n_local,
        MPI_Comm dim_ij_comm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double A[n_local*n_local], double BT[n_local*n_local],
        double* M, double* NT) 
{

    if(coords[2] == 0) {
        mpi_check(MPI_Scatterv(M, sendcounts,  displs, subarrtype,
                A, n_local*n_local, MPI_DOUBLE, root, dim_ij_comm));
    }


    //a small trick to transpose displs -> displsBT
    int displsBT[p_gridsize*p_gridsize];
    int i,j;
    if(root == me) {
        for(i = 0; i < p_gridsize; i++) {
            for(j = 0; j < p_gridsize; j++) {
                displsBT[i*p_gridsize+j] = displs[j*p_gridsize+i];
            }
        }
    }

    if(coords[2] == 0) {
        mpi_check(MPI_Scatterv(NT, sendcounts, displsBT, subarrtype,
                BT, n_local*n_local, MPI_DOUBLE, root, dim_ij_comm));
    }
}

void gather_result(int root, int me, int coords[3],
        int n, int p_gridsize, int n_local,
        MPI_Comm dim_ij_comm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double C[n_local*n_local],
        double* M, double* NT, double* P) 
{
    if(coords[2] == 0) 
    {
        MPI_Gatherv(
            C, n_local*n_local,  MPI_DOUBLE, P, 
            sendcounts, displs, subarrtype, root, dim_ij_comm
        );
    }

    //all in root now
    if (me == root) 
    {
        int err_c = check_result(M, NT, P, n, n, n, 0); // not transposed
        if (err_c) 
        {
            fprintf(stderr, "Check failed: %d errors\n", err_c);
        } else {
            printf("Result is correct\n");
        }
        mkl_free(M);
        mkl_free(NT);
        mkl_free(P);
    }
}

