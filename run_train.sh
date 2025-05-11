#!/bin/sh
#SBATCH --job-name=swe_gnn_train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440

. venv/bin/activate

echo "========== lrp01 =========="
#srun python main.py --config 'configs/lrp01_config.yaml'

echo "========== lrp04 =========="
srun python main.py --config 'configs/lrp04_config.yaml'

echo "========== lrp05 =========="
srun python main.py --config 'configs/lrp05_config.yaml'

echo "========== lrp08 =========="
srun python main.py --config 'configs/lrp08_config.yaml'