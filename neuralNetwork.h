/*
Here we define the neural network class and its functions
STB header only library can act simultaneously as a header file and implementation file for the class

#ifndef is used to avoid multiple inclusion of the same header file in the same translation unit (source file) which can cause compilation errors.

#define is used to define a macro which can be used to define constants or functions that can be used in the code.

#endif is used to end the conditional block started by #ifndef directive.

Since the entire approach with working on  neural networks revolves around matrices, so we need to have a trype that defines them.
We can use the typedef keyword to define a new type in C. The syntax for typedef is:

typedef existing_type new_type_name;

-> Matrices can be of any size, so we need to make them dynamic.
Also to properly define the matrix we can typdef a struct that contains the number of rows and columns and a pointer to the matrix elements.
The pointer to the matrix elements is of type float, so we can use it to store the matrix elements dynamically.    
But it means, we are working with the dybamic memory allocation, so we need to free the memory after we are done with the matrix.

The exanple of defining the matrix is shown below:
float d[] = {1, 2, 3, 4, 5, 6};
Mat m = {.rows = 2, .cols = 3, .es = d};

Where the .es = d is the pointer to the matrix elements, and the matrix is defined as a 2x3 matrix.

Suppose we have the training data for OR gate as follows:
float d[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1
};
Mat m = {.rows = 4, .cols = 2, .es = d}; here doing this we will not be able to keep track of when we have to change the row, so we have to introduce other parameter called stride.
Stride is the number of elements in the row, so we can define the matrix as follows:
Mat mi = {.rows = 4, .cols = 2, .stride = 3, .es = d}; here stride is 3, so we have 3 elements in each row.
Mat m0 = {.rows = 4, .cols = 1, .stride = 3, .es = d}; here we have 1 element in each row.

But as of now its not needed to define the stride, so we can define the matrix as follows:
Mat m = {.rows = 4, .cols = 3, .es = d};

So we create a operation that allocates memory for the matrix and returns a pointer to the matrix.

//two main operations are multiplication and addition of matrices
//multiplication of matrices is done by multiplying the elements of the row of the first matrix with the elements of the column of the second matrix
//I dont want to intake matrices and return a new matrix, for which we have to allocate the memory again and again,
//Instead I will take three matrices which will have the preallocated memory, and calculate the result in the third matrix
//wILL pass the third matrix as the furst argument, and the first two matrices as the second and third arguments, just similar to memcpy function

Also, we dont need to define the matrices as the pointers, as we are not going to allocate the memory for the matrices in the function, so we can define the matrices as the structures.
Already the structures are too light weight, so we can pass them by value, and not by reference.
They are merely 3 64 bit integers, so they can be passed by value.

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = malloc(sizeof(*m.es)*rows*cols);
    assert(m.es != NULL);
    return m;
}

This function will allocate the memory for the matrix and return the matrix.
1. Here the first line {Mat m;} creates a matrix structure.
2. The second line {m.rows = rows;} assigns the number of rows to the matrix.
3. The third line {m.cols = cols;} assigns the number of columns to the matrix.
4. The fourth line {m.es = malloc(sizeof(*m.es)*rows*cols);} allocates the memory for the matrix.
In the fourth line, specifically (*m.es) is used to get the size of the element of the matrix, and then we multiply it with the number of rows and columns to get the total size of the matrix.
We didnt use the sizeof(float) as the size of the float is not fixed, and it can vary from system to system, also if we change it from float to something else in the structure, then it will be a problem, so we use the sizeof(*m.es) to get the size of the element of the matrix.
5. The fifth line {assert(m.es != NULL);} checks if the memory is allocated or not, if not then it will throw an error.
assert is a macro that is used to check if the condition is true or not, if the condition is false, then it will throw an error and the program will stop.  
6. The sixth line {return m;} returns the matrix. 

//addition of matrices is done by adding the elements of the matrices element wise, and saving it in dest matrix.

FACT: On a 64 bit machine, the size of the pointer is 8 bytes, and the size of the structure is 24 bytes, so it is better to pass the structure by value, and not by reference.
Onr last operation that is needed is the printing of matrix.

Defining the own assert and malloc helps in the web assembly envrioment, where we can define our own malloc and assert functions, and use them in the code, and we dont have to use the lip c.

void mat_print(Mat m){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            printf("%f ", m.es[i*m.cols + j]);
        }
        printf("\n");
    }
}
Here:
1. The first line {void mat_print(Mat m){}} defines the function that prints the matrix.
2. The second line {for(size_t i = 0; i < m.rows; i++){}} defines the loop that iterates over the rows of the matrix.
3. The third line {for(size_t j = 0; j < m.cols; j++){}} defines the loop that iterates over the columns of the matrix.
4. The fourth line {printf("%f ", m.es[i*m.cols + j]);} prints the element of the matrix.
Specifically, m.es[i*m.cols + j] is used to get the element of the matrix, here i*m.cols is used to get the row, and j is used to get the column, which can be understood as
m.es[0*3 + 0] = m.es[0] = 1
m.es[0*3 + 1] = m.es[1] = 2
m.es[0*3 + 2] = m.es[2] = 3
m.es[1*3 + 0] = m.es[3] = 4
m.es[1*3 + 1] = m.es[4] = 5
m.es[1*3 + 2] = m.es[5] = 6
Which shows that by doing i*m.cols + j, we can get the element of the matrix.
5. The fifth line {printf("\n");} prints a new line after printing the row of the matrix.
But the m.es[i*m.cols + j] is not the best way to access the elements of the matrix, as it is not cache friendly, so we can use the stride to access the elements of the matrix.
Else we can define the matrix as a macra named MAT_AT for the m.es[i*m.cols + j] and use it to access the elements of the matrix.

As we will be using random numbers as input for the neural network, so we can use the rand() function to generate the random numbers.
For that also we need to define a function also.

We can use LAPACK for the matrix operations, but it is not available in the web assembly, so we have to define our own functions for the matrix operations.
lapack helps in doing the matrix operations in a very efficient way.
*/

//----------------------------------------CODE----------------------------------------  

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

//IN C, we can create our custom MALLOC function, which can be used to allocate the memory for the matrices.    
#ifndef NEURALNETWORK_MALLOC
#include <stdlib.h>
#define NEURALNETWORK_MALLOC malloc
#endif // NEURALNETWORK_MALLOC

//IN C, we can create our custom assert function, which can be used to check if the memory is allocated or not.
#ifndef MEURALNETWORK_ASSERT
#include <assert.h>
#define NEURALNETWORK_ASSERT assert
#endif // NEURALNETWORK_ASSERT

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]


float rand_float();
float sigmoidf(float x);

//allocating the memory to the matrix
Mat mat_alloc(size_t rows, size_t cols);
//ramdon number generator
void mat_rand(Mat m, float low, float high);
//multiplication of matrices;
void mat_dot(Mat dest, Mat a, Mat b);
//addition of matrices
void mat_sum(Mat dest, Mat a);
//printing the matrix
void mat_print(Mat m,const char *name, size_t padding);

#define MAT_PRINT(m) mat_print(m,#m,0)
//HERE #m is a stringizer, which converts the argument to a string
//So if we pass the matrix as MAT_PRINT(w1), then it will print the matrix as w1 = [....]

//matrix fill
void mat_fill(Mat m, float val);
//activating the matrix
void mat_sig(Mat m);
//matrix row 
Mat mat_row(Mat m, size_t row);
//copy the matrix
void mat_copy(Mat dest, Mat src);

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn,#nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN nn_g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN nn_g, float rate);

#endif // NEURALNETWORK_H_


//implementation of the class
#ifdef NEURALNETWORK_IMPLEMENTATION

float rand_float(){
    return (float)rand()/(float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NEURALNETWORK_MALLOC(sizeof(*m.es)*rows*cols);
    NEURALNETWORK_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat dest, Mat a, Mat b){
    NEURALNETWORK_ASSERT(dest.rows == a.rows);
    NEURALNETWORK_ASSERT(dest.cols == b.cols);
    NEURALNETWORK_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    for(size_t i=0; i<dest.rows; i++){
        for(size_t j=0; j<dest.cols; j++){
            MAT_AT(dest,i,j) = 0;
            for(size_t k=0; k<n; k++){
                MAT_AT(dest,i,j) += MAT_AT(a,i,k) * MAT_AT(b,k,j);
            }
        }
    }
}

void mat_sum(Mat dest, Mat a){
    //checking if the number of rows and columns of the matrices are same or not
    NEURALNETWORK_ASSERT(dest.rows == a.rows);
    NEURALNETWORK_ASSERT(dest.cols == a.cols);
    //iterating through
    for(size_t i=0; i<dest.rows; i++){
        for(size_t j=0; j<dest.cols; j++){
            MAT_AT(dest,i,j) += MAT_AT(a,i,j);
        }
    }
}

Mat mat_row(Mat m, size_t row){
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}

void mat_copy(Mat dest, Mat src){

    NEURALNETWORK_ASSERT(dest.rows==src.rows);
    NEURALNETWORK_ASSERT(dest.cols==src.cols);
    for(size_t i=0; i<dest.rows; i++){
        for(size_t j=0; j<dest.cols; j++){
            MAT_AT(dest,i,j) = MAT_AT(src,i,j);
        }
    }

}

void mat_print(Mat m, const char *name, size_t padding){
    printf("%*s%s = [\n", (int) padding , "", name);
    for(size_t i = 0; i < m.rows; i++){
        printf("%*s    ", (int) padding, "");
        for(size_t j = 0; j < m.cols; j++){
            printf("%f ", MAT_AT(m,i,j));
        }
        printf("%*s]\n", (int) padding, "");
    }
    printf("]\n");
}

void mat_rand(Mat m, float low, float high){
    for(size_t i=0; i<m.rows; i++){
        for(size_t j=0; j<m.cols; j++){
            MAT_AT(m,i,j) = rand_float()*(high - low) + low;
        }
    }
}

void mat_fill(Mat m, float val){
    for(size_t i=0; i<m.rows; i++){
        for(size_t j=0; j<m.cols; j++){
            MAT_AT(m,i,j) = val;
        }
    }
}

float sigmoidf(float x){
    return 1.0f/(1.0f+exp(-x));
}

void mat_sig(Mat m){
    for(size_t i=0; i<m.rows; i++){
        for(size_t j=0; j<m.cols; j++){
            MAT_AT(m,i,j) = sigmoidf(MAT_AT(m,i,j));
        }
    }
}

NN nn_alloc(size_t *arch, size_t arch_count){
    
    NEURALNETWORK_ASSERT(arch_count>0);
    //allocating the memory to the matrices.
    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NEURALNETWORK_MALLOC(sizeof(*nn.ws)*nn.count);
    NEURALNETWORK_ASSERT(nn.ws != NULL);
    nn.bs = NEURALNETWORK_MALLOC(sizeof(*nn.bs)*nn.count);
    NEURALNETWORK_ASSERT(nn.bs != NULL);    
    nn.as = NEURALNETWORK_MALLOC(sizeof(*nn.as)*(nn.count+1));
    NEURALNETWORK_ASSERT(nn.as != NULL); 

    //CONFIGURING THE FIRST INPUT LAYER

    nn.as[0] = mat_alloc(1, arch[0]);
    //allocating all of the other layers
    for(size_t i=0; i<nn.count; i++){
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1,arch[i]);
        nn.as[i] = mat_alloc(1,arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name){
    char buf[256];
    printf("%s = [\n",name);
    for(size_t i=0; i<nn.count; i++){
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high){
    for(size_t i=0; i<nn.count; i++){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn){
    for(size_t i=0; i<nn.count; i++){
         mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
         mat_sum(nn.as[i+1], nn.bs[i]);
         mat_sig(nn.as[i+1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to){
    NEURALNETWORK_ASSERT(ti.rows == to.rows);
    NEURALNETWORK_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    float c = 0; 
    for(size_t i=0; i<n; i++){
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q = to.cols;
        for(size_t j=0; j<q; j++){
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c+=d*d;
        }
    }
    return c/n;
}

void nn_finite_diff(NN nn, NN nn_g, float eps, Mat ti, Mat to){
    float saved;
    float c = nn_cost(nn,ti,to);
    for(size_t i = 0; i<nn.count; i++){
        for(size_t j=0; j<nn.ws[i].rows; j++){
            for(size_t k=0; k<nn.ws[i].cols; k++){
                saved = MAT_AT(nn.ws[i],j,k);
                MAT_AT(nn.ws[i], j, k) += eps;
                //store in the gradient
                MAT_AT(nn_g.ws[i],j,k) = (nn_cost(nn,ti,to)-c)/eps;
                //restore the saved value
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for(size_t j=0; j<nn.bs[i].rows; j++){
            for(size_t k=0; k<nn.bs[i].cols; k++){
                saved = MAT_AT(nn.bs[i],j,k);
                MAT_AT(nn.bs[i], j, k) += eps;
                //store in the gradient
                MAT_AT(nn_g.bs[i],j,k) = (nn_cost(nn,ti,to)-c)/eps;
                //restore the saved value
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN nn_g, float rate){
    for(size_t i = 0; i<nn.count; i++){
        for(size_t j=0; j<nn.ws[i].rows; j++){
            for(size_t k=0; k<nn.ws[i].cols; k++){
                MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(nn_g.ws[i], j, k);
            }
        }

        for(size_t j=0; j<nn.bs[i].rows; j++){
            for(size_t k=0; k<nn.bs[i].cols; k++){
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(nn_g.bs[i], j, k);
            }
        }
    }
}

#endif // NEURALNETWORK_IMPLEMENTATION