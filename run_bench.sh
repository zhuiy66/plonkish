#!/bin/bash  
  
# 要执行的次数  
n=$1  
  
# 检查是否提供了参数  
if [ -z "$n" ]; then  
    echo "Usage: $0 <number_of_iterations>"  
    exit 1  
fi  
  
# 循环n次  
for ((i=1; i<=$n; i++))  
do  
    # 执行cargo bench命令  
    cargo bench --bench proof_system -- --system hyperplonk --circuit vanilla_plonk --k 10..21 
  
    # 检查halo2文件是否存在，如果存在则重命名  
    if [ -f "./benchmark/bench/halo2" ]; then  
        mv "./benchmark/bench/halo2" "./benchmark/bench/halo2-verify-oneGate+oneLookup+oneCross-0502-${i}"  
    else  
        echo "File 'halo2' not found after iteration $i."  
    fi  
  
    # 检查hyperplonk文件是否存在，如果存在则重命名  
    if [ -f "./benchmark/bench/hyperplonk" ]; then  
        mv "./benchmark/bench/hyperplonk" "./benchmark/bench/hyperplonk-verify-oneGate+oneLookup+oneCross-0502-${i}"  
    else  
        echo "File 'hyperplonk' not found after iteration $i."  
    fi  
done  
  
echo "Script execution completed."