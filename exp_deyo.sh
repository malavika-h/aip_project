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
CMMODEL_NAME='/content/aip_project/pretrained/ColoredMNIST_model.pickle'  # Path to pretrained model

# MODEL=resnet18_bn
# python main.py --method $METHOD --data_root $ROOT --dset ColoredMNIST --gpu $GPU \
# --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR \
# --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --wandb_log $LOG \
# --cmmodel_name $CMMODEL_NAME

# Run till it downloads the dataset, then stop and comment out
# python pretrain_Waterbirds.py --root_dir $ROOT --dset_dir Waterbirds --gpu $GPU --seed $SEED

LRMUL=5
DTHR=0.5
MODEL=resnet50_bn_torch
INTERVAL=10

WBMODEL_NAME='/content/aip_project/pretrained/waterbirds_pretrained_model.pickle'
python main.py --method $METHOD --data_root $ROOT --dset Waterbirds --gpu $GPU \
--wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR \
--lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --wandb_log $LOG \
--wbmodel_name $WBMODEL_NAME