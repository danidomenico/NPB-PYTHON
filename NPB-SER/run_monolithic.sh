#!/bin/bash

BACKENDS=("numba" "scipy" "petsc" "torch")
RESULTS_DIR="results_mono"
NITER=5
mkdir -p "$RESULTS_DIR"

if [ "$#" -eq 0 ]; then
    ARGS=("S")
else
    ARGS=("$@")
fi

CLASSES=()
for arg in "${ARGS[@]}"; do
    for (( idx=0; idx<${#arg}; idx++ )); do
        CLASSES+=("${arg:$idx:1}")
    done
done

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu

for CLASS in "${CLASSES[@]}"; do
    for BACKEND in "${BACKENDS[@]}"; do
        OUTPUT_FILE="${RESULTS_DIR}/${BACKEND}_${CLASS}.txt"
        
        echo "Iniciando $NITER execuções para: $BACKEND (Classe $CLASS)"
        > $OUTPUT_FILE # limpa o arquivo de resultados

        for i in {1..$NITER}; do
            taskset -c 1 python CG_Monolithic.py -c $CLASS -b $BACKEND | grep "Time in seconds" >> $OUTPUT_FILE 
            echo "    [$BACKEND] Execução $i concluída."
            sleep 3
        done
        
        echo "Testes do $BACKEND finalizados. Arquivo gerado: $OUTPUT_FILE"
        echo ""
        
        sleep 3
    done
done
