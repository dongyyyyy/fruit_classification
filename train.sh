#!/bin/bash

echo
{
 python main.py -mode test -model efficientnet-b0 -optim sgd -lr 0.01 -scheduler multiStepLR -batch_size 64 -size 608 -random_seed 0 -loss_function CE -fold 5;
 python main.py -mode test -model efficientnet-b0 -optim sgd -lr 0.01 -scheduler multiStepLR -batch_size 64 -size 608 -random_seed 0 -loss_function CE -fold 0;
#  python main.py -mode train -model efficientnet-b0 -optim sgd -lr 0.01 -scheduler multiStepLR -batch_size 64 -size 608 -random_seed 0 -loss_function CE -fold 5;
#  python main.py -mode train -model efficientnet-b0 -optim sgd -lr 0.01 -scheduler multiStepLR -batch_size 64 -size 608 -random_seed 1 -loss_function CE -fold 5;
#  python main.py -mode train -model efficientnet-b0 -optim sgd -lr 0.01 -scheduler multiStepLR -batch_size 64 -size 608 -random_seed 2 -loss_function CE -fold 5;
#  python main.py -mode train -model efficientnet-b0 -optim sgd -lr 0.01 -scheduler multiStepLR -batch_size 64 -size 608 -random_seed 3 -loss_function CE -fold 5;
}
