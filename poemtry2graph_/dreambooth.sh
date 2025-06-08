#!/bin/bash

#SBATCH --job-name=dreambooth # 作业名

#SBATCH --partition=A800 # A800 队列

#SBATCH -N 1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=4 # 1:4 的 GPU:CPU 配比

#SBATCH --gres=gpu:a800:1 # 1 块 GP

                

#SBATCH --output=%j.out

#SBATCH --error=%j.err


echo "作业开始，Job ID: $SLURM_JOB_ID"
echo "加载 Conda 环境变量..."
# 加载conda环境变量
source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh

echo "激活 Conda 环境: diffuser_env..."
# 激活你创建的虚拟环境
conda activate diffuser_env

# 检查当前 python 和 pip 路径，确保环境正确
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

echo "开始执行训练脚本 run_training.sh..."
# 执行我们刚刚创建的训练脚本
# 使用 bash 执行，而不是 srun，因为 accelerate launch 会自己处理分布式启动
bash ./run_training.sh

echo "作业结束"