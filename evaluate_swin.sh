name='lip_solider_swin_tiny'
python evaluate.py --arch swin_tiny --data-dir /home/ubuntu/PaddleSeg/ --model-restore ./logs/${name}/schp_4_checkpoint.pth.tar --input-size 572,384 --multi-scales 0.5,0.75,1.0,1.25,1.5 --flip --batch-size 1 --save-results
