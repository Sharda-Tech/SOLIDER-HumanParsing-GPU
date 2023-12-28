# tiny
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 8 train.py --arch swin_tiny --imagenet-pretrain ./pretrain/solider_swin_tiny.pth --batch-size 2  --learning-rate 7e-3 --weight-decay 0 --optimizer sgd --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 150 --schp-start 120 --input-size 572,384 --log-dir ./logs/lip_solider_swin_tiny
