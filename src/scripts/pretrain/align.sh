python src/models/selfsupervised/trainers/align_trainer.py \
    --device $CUDA_VISIBLE_DEVICES \
    --vision-backbone resnet34 \
    --job_number ${SLURM_JOBID} \
    --file_name ALIGN-${SLURM_JOBID} \
    --epochs 300 \
    --transforms_cxr simclrv2 \
    --temperature 0.01 \
    --batch_size 256 \
    --lr 0.00006026 \
    --num_gpu 1 \
    --pretrain_type simclr \
    --mode train \
    --fusion_type None \
    --pairs_csv MedMod/medmod_pairs.csv \
    --medmod_pairs_path medmod_pairs.csv \
    --tag align_train_phenotyping \
    --save_dir MedMod/src/checkpoints/pretrain/models \
    --tmp_dir /home/user/fork/CS598

