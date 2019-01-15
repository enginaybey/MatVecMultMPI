/*
 ============================================================================
 Name        : cart_mult.c
 Author      : Engin Aybey
 Version     : 1.0.0
 Copyright   : All rights reserved.
 Description : Virtual Topology Implementation with MPI
               Matrix-Vector Multiplication
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define TRUE 1
#define FALSE 0

int main(int argc, char *argv[])
{
    int rank, size;
    int newrank, newsize;
    int n=8; // size of matrix and vector
    double start_time;
    double end_time;
    double tot,time;
    MPI_Comm new_comm,commrow,commcol;
    MPI_Datatype newtype;
    MPI_Datatype newtype2;
    MPI_Datatype vtype;
    int dim[2], period[2], reorder;
    int coord[2], id, ndims;
    int belongs[2];
    int colrank,rowrank;
    int i,j,t;
    float *A_array;
    float *b;
    //float *C;
    //float **A;
    float *sub_block_1;
    float *sub_block_2;
    float *vpart;
    float *C_temp;
    float *C_result;
    float *C_buf;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    MPI_Request request;
    int s_root_procs=(int)sqrt((double)size);
    int num=(int)((double)n/(double)s_root_procs);

    sub_block_1 = (float *) malloc (num*n * sizeof(float));
    sub_block_2 = (float *) malloc (((n*n)/size) * sizeof(float));
    vpart = (float *) malloc(num * sizeof(float));
    C_temp = (float*)malloc(num*sizeof(float));
    C_result = (float*)malloc(n*sizeof(float));
    C_buf = (float*)malloc(num*sizeof(float));
    MPI_Type_contiguous(num*n,MPI_FLOAT,&newtype);
    MPI_Type_contiguous(num,MPI_FLOAT,&newtype2);
    MPI_Type_vector(num,num,n,MPI_FLOAT,&vtype);
    MPI_Type_commit(&newtype);
    MPI_Type_commit(&newtype2);
    MPI_Type_commit(&vtype);

    if(rank==0){
     // store A matrix in A_array vector
//    t=0;
//    A = (float **)malloc(n*sizeof(float *));
    A_array = (float*)malloc(n*n*sizeof(float));
    for(i=0 ;i<n; i++){
//            A[i] = (float *)malloc(n*sizeof(float));
            for(j=0; j<n; j++) {
//                 A[i][j]=t;
                 A_array[i*n+j]=(rand() / (float)RAND_MAX);
//                 t++;
            }
    }
    // construct b vector
    b = (float*)malloc(n*sizeof(float));
//    C = (float*)malloc(n*sizeof(float));
    for(i = 0; i<n; i++) {
       b[i]=(rand() / (float)RAND_MAX);
//       C[i]=0.0;
    }

/*    
    for(i=0 ;i<n; i++){
            for(j=0; j<n; j++) {
               C[i]+=A[i][j]*b[j]; 
            }
    }
    for(i=0 ;i<n; i++){
       printf("%f ",C[i]);
    }
*/
    }

    if ( size != 4&&size!=16&&size!=64 ){
        fprintf(stdout,"Please run the program with 4 or 16 or 64 CPUs!!\n");
        MPI_Finalize();
        exit(1);
    }
    // Create a 2D cartesian topology
    ndims=2;            /*  2D matrix grid */
    dim[0]=s_root_procs;           /* rows */
    dim[1]=s_root_procs;           /* columns */
    period[0]=TRUE;     /* row periodic (each column forms a ring) */
    period[1]=FALSE;    /* columns nonperiodic */
    reorder=TRUE;       /* allows processes reordered for efficiency */
    MPI_Cart_create(MPI_COMM_WORLD,ndims,dim,period,reorder,&new_comm);

    MPI_Cart_coords(new_comm, rank, ndims, coord);
    MPI_Cart_rank(new_comm,coord, &newrank);
//    fprintf(stdout,"%d - Coordinates -->> %d %d\n",rank,coord[0],coord[1]);
//    fflush(stdout);

//    MPI_Barrier(new_comm);
/* Create 1D row subgrid */
    belongs[0]=0;
    belongs[1]=1; /* this dimension belongs to subgrid */
//    if(newrank == 0)
//        fprintf(stdout,"\nRank\t(1D iD)\t(x,y)\tR/C\n\n");fflush(stdout);

    MPI_Cart_sub(new_comm, belongs, &commrow);
    MPI_Comm_rank(commrow,&rowrank);
//    fprintf(stdout,"%3d\t%4d\t(%d,%d)\tR\n",newrank, rowrank, coord[0],coord[1]);fflush(stdout);

//    MPI_Barrier(MPI_COMM_WORLD);

    /* Create 1D column subgrids */
    belongs[0]=1; /* this dimension belongs to subgrid */
    belongs[1]=0;

    MPI_Cart_sub(new_comm, belongs, &commcol);
    MPI_Comm_rank(commcol,&colrank);
//    fprintf(stdout,"%3d\t%4d\t(%d,%d)\tC\n",newrank,colrank, coord[0],coord[1]);fflush(stdout);
//    fflush(stdout);
    for(i = 0; i<num; i++) {
        C_temp[i]=0.0;
        C_buf[i]=0.0;
    }
    // Partition the A_array to first column processors 
//    MPI_Barrier(new_comm);
    if(rowrank==0){
       MPI_Scatter(A_array, 1, newtype, sub_block_1, num*n, MPI_FLOAT, 0, commcol);
    }
    // Partition the b vector to first row processors 
    if(colrank==0){
       MPI_Scatter(b, 1, newtype2, vpart, num, MPI_FLOAT, 0, commrow);
    }
    // Broadcast the parts of b vector from the masters of the columns to their slaves
    MPI_Bcast(vpart, num, MPI_FLOAT, 0, commcol);

    // Partition the parts of the A_array in each processors in first column to the processors in related rows by using vector type 
//    MPI_Barrier(new_comm);
    if (rowrank==0)
    {
        for(i=0;i<s_root_procs;i++) MPI_Isend(&sub_block_1[i*num], 1, vtype, i, i, commrow,&request);
    }
//    MPI_Barrier(new_comm);
    if (rowrank>=0)
    {
        MPI_Recv(sub_block_2, ((n*n)/size), MPI_FLOAT, 0, rowrank, commrow, &status);
//        for (i=0; i<4; i++)
//            printf("%d, %d-buff[%d] = %f\n", newrank,rowrank,i,Block[i]);
//        fflush(stdout);
    }
    //Multiply sub blocks of A_array by vector parts in each processors
//    MPI_Barrier(new_comm);
    start_time = MPI_Wtime();
    for(i=0;i<num;i++){
       for(j=0;j<num;j++){
           C_temp[i]+=sub_block_2[i*num+j]*vpart[j];
       }
    }
    end_time = MPI_Wtime();
    time=(double)(end_time - start_time);
//    printf("\n--Running Time for %d X %d = %f\n\n",n,n,end_time - start_time);
    MPI_Reduce(&time,&tot,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank==0) printf("\nRunning Time for %d X %d = %f\n\n",n,n,tot/(double)size);
    //Sum the parts of the multiplication results from each processors to masters of the rows
//    MPI_Barrier(new_comm);
    MPI_Reduce(C_temp,C_buf,num,MPI_FLOAT,MPI_SUM,0,commrow);
    // Gather the results from the first column processors to the master of the first column processors,namely rank=0 in MPI_COMM_WORLD.
    if(rowrank==0){
       MPI_Gather(C_buf,num,MPI_FLOAT,C_result,num,MPI_FLOAT,0,commcol);
    }

    // look at the results    
//    MPI_Barrier(new_comm);

    if(newrank==0){
    for(i=0 ;i<n; i++){
       printf("%f ",C_result[i]);
    }
    printf(" \n");
    }

    MPI_Finalize();

    return 0;
}


