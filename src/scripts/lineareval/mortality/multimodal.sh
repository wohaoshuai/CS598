

source activate MedMod

python MedMod/src/models/selfsupervised/trainers/finetune.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 200 --batch_size 256 --lr 0.001 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--load_state MedMod/src/checkpoints/pretrain/models/SIMCLR-8538298/SIMCLR-8538298_epoch_65.ckpt \
--file_name SIMCLR-8538298-E65-FT-${SLURM_JOBID} \
--finetune \
--pretrain_type simclr \
--fusion_type lineareval \
--mode train \
--save_dir MedMod/checkpoints/finetune/mortality/models \
--tag finetuning_train_mortality  \