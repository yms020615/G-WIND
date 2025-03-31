python train_model_gwind.py \
    --epochs 100 \
    --batch_size 16 \
    --kl_beta 1e-2 \
    --lr 1e-3

# python train_model_gwind.py \
    # --epochs 7 \
    # --batch_size 4 \
    # --kl_beta 1e-2 \
    # --lr 1e-4 \
    # --ar_steps 8 \
    # --load # .ckpt

# python train_model_gwind.py \
    # --epochs 100 \
    # --batch_size 32 \
    # --kl_beta 1e-2 \
    # --lr 1e-4 \
    # --load # .ckpt

# python train_model_gwind.py \
    # --epochs 1 \
    # --batch_size 8 \
    # --kl_beta 1e-2 \
    # --lr 1e-5 \
    # --crps_weight 0.001 \
    #--load # .ckpt