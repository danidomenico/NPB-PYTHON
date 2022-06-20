#DIR="~/Documentos/Doutorado/docker_gpu/apps/NPB-Perf"
DIR="~/apps/NPB-Perf"

REP=30

COMMANDS=()
I=0

#CPP
EXEC_CPP_SER="$DIR/NPB-CPP-SER/bin/"
EXEC_CPP_CUDA="$DIR/NPB-CPP-CUDA/bin/"

#PYTHON
EXEC_PYT_SER="python $DIR/NPB-PYTHON-SER/"
EXEC_PYT_CUDA="python $DIR/NPB-PYTHON-CUDA/"

#OPENACC
EXEC_OPENACC="ulimit -s unlimited && $DIR/NPB-CPP-OPENACC/PGI/BT/bin/"

BENCHS=("CG" "EP" "FT" "IS" "MG")
#BENCHS=("BT" "LU" "SP")
CLASSES=("B" "C")

for b in ${BENCHS[*]}; do
	echo "Bench: $b"
	b_low=$(echo $b | awk '{print tolower($0)}')
	echo "Bench Low: $b_low"
	for c in ${CLASSES[*]}; do
		echo "Class: $c"
		
		for ((i=0;i<$REP;i++)); do
			COMMANDS[I]="$EXEC_CPP_SER$b_low.$c"
			I=$((I+1))
			
			COMMANDS[I]="$EXEC_CPP_CUDA$b_low.$c"
			I=$((I+1))
			
			COMMANDS[I]="$EXEC_PYT_SER$b.py -c $c"
			I=$((I+1))
			
			COMMANDS[I]="$EXEC_PYT_CUDA$b.py -c $c"
			I=$((I+1))
			
			if [ $b != 'IS' ]; then
				COMMANDS[I]="$EXEC_OPENACC$b_low.$c.x"
				I=$((I+1))
			fi
		done
	done
done

ARRAY_SIZE=${#COMMANDS[@]}
ARRAY_LIMIT=$((ARRAY_SIZE-1))

echo $ARRAY_SIZE $ARRAY_LIMIT

ARRAY_INDEX=( $(shuf -i 0-$ARRAY_LIMIT -n $ARRAY_SIZE) )

N=0

#Exec
for i in ${ARRAY_INDEX[*]}; do
	#echo $i
	echo $N
	N=$((N+1))
	eval "${COMMANDS[i]}"
done

#Validate
#for ((i=0; i<$ARRAY_SIZE; i++)); do
#	echo "${ARRAY_INDEX[i]} ${COMMANDS[i]}"
#done
