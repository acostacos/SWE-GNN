#!/bin/sh
#SBATCH --job-name=swe_gnn_validate
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000

. venv/bin/activate

echo "========== initp01 =========="
srun python validate.py --config 'configs/initp01_config.yaml' --output_path 'saved_metrics/SWEGNN_initp01_metrics.npz' --model_path 'wandb/initp01-train-0513-20epoch/files/339i9547.h5'

echo "========== lrp01 =========="
srun python validate.py --config 'configs/lrp01_config.yaml' --output_path 'saved_metrics/SWEGNN_lrp01_metrics.npz' --model_path 'wandb/lrp01-train-0514/files/b2uwdqra.h5'

#echo "========== lrp04 =========="

#echo "========== lrp05 =========="

#echo "========== lrp08 =========="

echo "========== HydroGraphNet =========="
srun python validate.py --config 'configs/hydrographnet_config.yaml' --model_path 'wandb/hydrographnet-train-0513/files/a4x4w67x.h5' --output_path 'saved_metrics/SWEGNN_H401_metrics.npz' 'saved_metrics/SWEGNN_H402_metrics.npz' 'saved_metrics/SWEGNN_H403_metrics.npz' 'saved_metrics/SWEGNN_H404_metrics.npz' 'saved_metrics/SWEGNN_H405_metrics.npz' 'saved_metrics/SWEGNN_H406_metrics.npz' 'saved_metrics/SWEGNN_H407_metrics.npz' 'saved_metrics/SWEGNN_H408_metrics.npz' 'saved_metrics/SWEGNN_H409_metrics.npz' 'saved_metrics/SWEGNN_H410_metrics.npz'

echo "========== Orig Data =========="
srun python validate.py --output_path 'saved_metrics/orig.npz' --model_path 'wandb/orig-data-train-0506/files/x66t1n8r.h5'
 srun python validate.py --output_path 'saved_metrics/GCN_test_metrics.npz' --model_path 'wandb/orig-data-gcn-train-0512/files/8i7ngh1m.h5'
