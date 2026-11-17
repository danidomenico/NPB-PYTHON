#!/bin/bash
BACKEND="numba"
CLASS="A"
OUTPUT_FILE="resultados_${BACKEND}_${CLASS}.txt"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export JAX_PLATFORMS=cpu

echo "Iniciando 30 execuções isoladas no Núcleo 0 para $BACKEND Classe $CLASS..."
> $OUTPUT_FILE 

for i in {1..30}; do
    taskset -c 0 python CG_Modulate.py -c $CLASS -b $BACKEND | grep "Time in seconds" >> $OUTPUT_FILE
    
    echo "Execução $i concluída."
    
    sleep 5
done