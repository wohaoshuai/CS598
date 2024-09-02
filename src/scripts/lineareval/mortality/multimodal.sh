#!/bin/bash
#SBATCH -c 5
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name ft_train_mortality
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/finetune/mortality/ft_train_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/finetune/mortality/ft_train_%j.err

source activate mml-ssl

python /scratch/se1525/MedMod/src/models/selfsupervised/trainers/finetune.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 200 --batch_size 256 --lr 0.001 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--load_state /scratch/se1525/MedMod/src/checkpoints/pretrain/models/SIMCLR-8538298/SIMCLR-8538298_epoch_65.ckpt \
--file_name SIMCLR-8538298-E65-FT-${SLURM_JOBID} \
--finetune \
--pretrain_type simclr \
--fusion_type lineareval \
--mode train \
--save_dir /scratch/se1525/MedMod/checkpoints/finetune/mortality/models \
--tag finetuning_train_mortality  \