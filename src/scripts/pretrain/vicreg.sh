

source activate MedMod

python MedMod/src/models/selfsupervised/trainers/vicreg_trainer.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--job_number ${SLURM_JOBID} \
--file_name VICREG-${SLURM_JOBID} \
--epochs 300 --transforms_cxr simclrv2 --temperature 0.01 \
--vicreg \
--num_gpu 1 \
--batch_size 256 --lr 0.5989 \
--pretrain_type simclr \
--mode train \
--fusion_type None \
--save_dir MedMod/src/checkpoints/pretrain/models  \
--tag vicreg_train_phenotyping \