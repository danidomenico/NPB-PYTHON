#!/bin/bash
CLASS="B"

# Lista com o nome exato dos arquivos dos seus backends (sem o .py)
BACKENDS=("numba" "scipy" "petsc" "jax" "torch")

# BLOQUEIO DE MULTITHREADING OCULTO
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu

for BACKEND in "${BACKENDS[@]}"; do
    OUTPUT_FILE="resultados_${BACKEND}_${CLASS}.txt"
    
    echo "======================================================"
    echo "Iniciando 30 execuções para: $BACKEND (Classe $CLASS)"
    echo "======================================================"
    > $OUTPUT_FILE # Limpa o arquivo de resultados desse backend

    for i in {1..5}; do
        # Isola no núcleo 0, roda e filtra o tempo
        taskset -c 0 python CG_modulate.py -c $CLASS -b $BACKEND | grep "Time in seconds" >> $OUTPUT_FILE
        
        echo "  [$BACKEND] Execução $i concluída."
        
        # Respiro térmico para o processador
        sleep 2
    done
    
    echo "-> Testes do $BACKEND finalizados. Arquivo gerado: $OUTPUT_FILE"
    echo ""
    
    # Respiro maior entre a troca de backends (30 segundos)
    sleep 5
done

echo "Todos os backends foram testados com sucesso!"