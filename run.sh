#!/bin/bash

# Once for 1 thread
echo "--------------------------------------- 1 THREADS"
export OMP_NUM_THREADS=1
for i in 16 32 64 96 128 150 180 200
do
	echo "$i number of cells"
	echo `build/kdtree_bench $i` >> "kdtree_1t.txt"
done
echo ""
echo ""


for t in `seq 2 2 20`
do
	export OMP_NUM_THREADS=${t}
	echo "--------------------------------------- ${t} THREADS"
	for i in 16 32 64 96 128 150 180 200
	do
		echo "$i number of cells"
		echo `taskset -c 0-9,20-29 build/kdtree_bench $i` >> "kdtree_${t}t.txt"
	done
	echo ""
	echo ""
done

