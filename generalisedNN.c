#define NEURALNETWORK_IMPLEMENTATION
#include "../Day2/neuralNetwork.h"
#include <time.h>

//this is the training data
float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

int main(void){
    //we are implementing xor gate
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    //we are slicing the ttraining data into two matrices, input and output matrices respectively
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n, 
        .cols = 1, 
        .stride = stride,
        .es = td + 2
    };
    
    //defininf the architecture of the neural network.
    //2 input layers, 2 hidden layers and one output layers.
    size_t arch[] = {2,2,1};
    //allocated the memory to the model.
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN nn_g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn,0,1);

    float eps = 1e-3;
    float rate = 1e-1;

    //trained using the dumb approach of finite differences.
    printf("cost = %f\n", nn_cost(nn, ti, to));
    for(size_t i=0; i<10*1000; i++){
        nn_finite_diff(nn, nn_g, eps, ti, to);
        nn_learn(nn, nn_g, rate);
        printf("%zu: cost = %f\n", i, nn_cost(nn, ti, to));
    }

    //ultimately it brings the cost function down;

    return 0;
}