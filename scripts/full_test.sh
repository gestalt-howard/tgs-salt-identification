# Shell script for creating test predictions in full mode

trn_path=../data/train_images/
msk_path=../data/train_masks/
tst_path=../data/test_images/

#### IMPORTANT! Make sure tgs_pred reflects these values####
mod_path=../weights/model_res19/
val_flag=1
prd_flag=0
mod_name=epoch_15N.pth
############################################################

batch_size=20
NUM_TRAIN=3600
NUM_FULL=4000

python2 ../tgs_pred.py --trn_path=$trn_path --msk_path=$msk_path --tst_path=$tst_path --mod_path=$mod_path --val_flag=$val_flag --prd_flag=$prd_flag --mod_name=$mod_name --batch_size=$batch_size --NUM_TRAIN=$NUM_TRAIN --NUM_FULL=$NUM_FULL
