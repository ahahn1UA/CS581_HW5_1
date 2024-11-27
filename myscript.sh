#!/bin/bash

source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.7.0

nvcc HW5.cu -o HW5
./HW5 5000 5000
./HW5 5000 5000
./HW5 5000 5000
./HW5 10000 5000
./HW5 10000 5000
./HW5 10000 5000
