


source activate MedMod

python MedMod/src/models/selfsupervised/trainers/model_selection.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 100 --batch_size 256 --lr 0.01 \
--job_number ${SLURM_JOBID} \
--load_state SIMCLR-8538298 \
--file_name SIMCLR-8538298-select-LC \
--width 1 \
--task phenotyping \
--labels_set pheno \
--pretrain_type simclr \
--fusion_type lineareval \
--mode train \
--fusion_layer 0 \
--save_dir mml-ssl/checkpoints/model_selection/models \
--tag model_selection \