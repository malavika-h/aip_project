GPU=0
LOG=0
METHOD=deyo
ETHR=0.5
EMAR=0.4
DTHR=0.3
INTERVAL=100
SEED=2024
LRMUL=5

ROOT='/content/aip_project/data'  # Modify to your data root directory
MODEL=vitbase_timm
# Mix_shifts setting
EXP=mix_shifts
INTERVAL=100
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

# Label_shifts setting
EXP=label_shifts
INTERVAL=100
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

# bs1 setting
EXP=bs1
INTERVAL=10000
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG
