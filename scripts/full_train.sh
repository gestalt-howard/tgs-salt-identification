# Shell script for training models in full mode

trn_path=../data/train_images/
msk_path=../data/train_masks/
tst_path=../data/test_images/
mod_path=../weights/model_0/

batch_size=20
epochs=300
starting_epoch=1

print_every=4
NUM_TRAIN=3600
NUM_FULL=4000

python2 ../tgs_main.py --trn_path=$trn_path --msk_path=$msk_path --tst_path=$tst_path --mod_path=$mod_path --batch_size=$batch_size --epochs=$epochs --starting_epoch=$starting_epoch --print_every=$print_every --NUM_TRAIN=$NUM_TRAIN --NUM_FULL=$NUM_FULL
