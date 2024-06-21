/**********************************************************
 *                                                        *
 *                Author: Keshav(@masterK0927)            *
 *                Started: 05/06/2024                     *
 *                Last Edited: 21st June, 24              *
 *                                                        *
 **********************************************************/


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
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_zero(NN nn);

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


void nn_backprop(NN nn, NN g, Mat ti, Mat to){
    NEURALNETWORK_ASSERT(ti.rows == to.rows);
    NEURALNETWORK_ASSERT(NN_OUTPUT(nn).cols == to.cols);
    size_t n = ti.rows;
    nn_zero(g);

    //i - current sample
    //l - current layer
    //j - current activation
    //k - previous activation

    for(size_t i=0; i<n; i++){
        mat_copy(NN_INPUT(nn), mat_row(ti,i));
        nn_forward(nn);
        for(size_t j = 0; j<=nn.count; i++){
            mat_fill(g.as[j],0);
        }
        for(size_t j=0; j<n; j++){
            MAT_AT(NN_OUTPUT(g),0,j) = MAT_AT(NN_OUTPUT(nn),0,j) - MAT_AT(to, i, j);
        }
        for(size_t l = nn.count; l>0; l--){
            for(size_t j=0; j<nn.as[l].cols; j++){
                float a = MAT_AT(nn.as[l],0,j);
                float da = MAT_AT(g.as[l], 0, j);
                MAT_AT(g.bs[l-1],0,j) += 2*da*a*(1-a);
                //iterating all the previous activations
                for(size_t k=0; k<nn.as[l-1].cols; k++){

                    // j = weight matrix col
                    // k = weight matrix row

                    float pa = MAT_AT(nn.as[l-1],0,k);
                    float w = MAT_AT(nn.ws[l-1],k,j);
                    //calculating the partial derivative with activation from prev layer
                    MAT_AT(g.ws[l-1],k,j)+=2*da*a*(1-a)*pa;
                    //iterating to the previous layer from current layer
                    MAT_AT(g.as[l-1],0,k) += 2*da*a*(1-a)*w;
                }
            }
        }
    }
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

void nn_zero(NN nn){
    for(size_t i=0; i<nn.count; i++){
        mat_fill(nn.ws[i],0);
        mat_fill(nn.bs[i],0);
        mat_fill(nn.as[i],0);
    }
    mat_fill(nn.as[nn.count],0);
}

#endif // NEURALNETWORK_IMPLEMENTATION