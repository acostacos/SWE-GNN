#!/bin/sh
#SBATCH --job-name=swe_gnn_validate
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000

. venv/bin/activate

echo "========== lrp01 =========="
srun python validate.py --config 'configs/lrp01_config.yaml' --output_path 'saved_metrics/SWEGNN_lrp01_metrics.npz' --model_path 'wandb/lrp01-train-0510/files/6qq9nhr0.h5'

#echo "========== lrp04 =========="

#echo "========== lrp05 =========="

#echo "========== lrp08 =========="