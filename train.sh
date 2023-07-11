### single node training scripts

# single gpu
CUDA_VISIBLE_DEVICES=0, \
    python main.py \
        --base configs/latent-diffusion/affordance.yaml \
        -t \
        --gpus 0,  \
        --scale_lr False \
        # --debug

# multi gpu
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,  \
#     python main.py \
#         --base configs/latent-diffusion/affordance.yaml \
#         -t \
#         --gpus 0,1,2,3,4,5,6,7,  \
#         --scale_lr False \
