# Shell script for running postprocessing.py

# Flag for turning on/off unit testing mode
uni_flag=0

# Data path to where predictions are stored. Typical format:
# ../weights/MODEL_FOLDER(SPECIFY)/preds/
dat_path=../weights/model_res19/validate/val_preds/

# These should not need to be changed
img_name=pred_data.h5
nme_name=pred_name.pickle

# Change submission name as desired
sub_name=submit.csv

# Change threshold according to submission results
thres=0.5

python2 postprocessor.py --uni_flag=$uni_flag --dat_path=$dat_path --img_name=$img_name --nme_name=$nme_name --sub_name=$sub_name --thres=$thres
