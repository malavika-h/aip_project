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

MODEL=resnet18_bn
python main.py --method $METHOD --data_root $ROOT --dset ColoredMNIST --gpu $GPU \
--wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR \
--lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --wandb_log $LOG \
--cmmodel_name $CMMODEL_NAME