# Shell script for training models in debug mode

trn_path=../data/debug_train/
msk_path=../data/debug_masks/
tst_path=../data/debug_test/
mod_path=../weights/model_tmp/

epochs=10

python2 ../tgs_main.py --trn_path=$trn_path --msk_path=$msk_path --tst_path=$tst_path --mod_path=$mod_path --epochs=$epochs
