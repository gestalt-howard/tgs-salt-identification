# Shell script for reformatting PyTorch .pth weight files

mod_path=../weights/model_res19R/
mod_name=epoch_66.pth
sav_name=epoch_66N.pth

python2 weight_reformatter.py --mod_path=$mod_path --mod_name=$mod_name --sav_name=$sav_name
