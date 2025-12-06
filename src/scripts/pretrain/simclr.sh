



python src/models/selfsupervised/trainers/vicreg_trainer.py \
    --device 0 \
    --vision-backbone resnet34 \
    --job_number 123456 \
    --file_name 123456 \
    --epochs 300 \
    --transforms_cxr simclrv2 \
    --temperature 0.01 \
    --vicreg \
    --num_gpu 1 \
    --batch_size 5 \
    --lr 0.5989 \
    --pretrain_type simclr \
    --mode train \
    --fusion_type None \
    --save_dir MedMod/src/checkpoints/pretrain/models
