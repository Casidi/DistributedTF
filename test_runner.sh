#!/bin/bash

rm -f test_results.txt

for n in 2 3 4 5
do
    for pop_size in 10 20 30 40 50
    do
    echo n = $n pop_size = $pop_size
    #mpirun --oversubscribe -n $n python main_manager.py $pop_size
    done
done

#mpirun --oversubscribe -n 2 python main_manager.py 20
#mpirun --oversubscribe -n 3 python main_manager.py 40
#mpirun --oversubscribe -n 3 python main_manager.py 50
#mpirun --oversubscribe -n 4 python main_manager.py 10
#mpirun --oversubscribe -n 4 python main_manager.py 30
#mpirun --oversubscribe -n 4 python main_manager.py 40
#mpirun --oversubscribe -n 4 python main_manager.py 50
#mpirun --oversubscribe -n 5 python main_manager.py 20
#mpirun --oversubscribe -n 5 python main_manager.py 30
mpirun --oversubscribe -n 5 python main_manager.py 40
#mpirun --oversubscribe -n 5 python main_manager.py 50
