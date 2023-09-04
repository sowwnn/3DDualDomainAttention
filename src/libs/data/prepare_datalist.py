# %%writefile /kaggle/working/BrainTumour_Seg/libs/data/prepare_datalist.py

import argparse
import glob
import json
import os
import pandas as pd
import torch

import monai
from sklearn.model_selection import train_test_split

def produce_sample_dict(line, stage):
    names = os.listdir(line)
    seg, t1ce, t1, t2, flair = [], [], [], [], []
    for name in names:
        name = os.path.join(line, name)
        
        if "_seg.nii" in name:
            seg.append(name)
        elif "_t1ce.nii" in name:
            t1ce.append(name)
        elif "_t1.nii" in name:
            t1.append(name)
        elif "_t2.nii" in name:
            t2.append(name)
        elif "_flair.nii" in name:
            flair.append(name)
    if stage == 'test':
#         print(f'{"image": t1ce + t1 + t2 + flair}')
        return {"image": t1ce + t1 + t2 + flair}
    else:
        try:
            return {"label": seg[0], "image": t1ce + t1 + t2 + flair}
        except: pass

def produce_sample_dict_csv(line, stage, csv= None, clas=False):

    names = os.listdir(line)
    pid = line.split('/')[-2]
    sur = None
    cl = None
    try:
        sur = csv.loc[pid]["Survival"]
        sur = sur * 1.0
        if clas:
            if sur >= 450:
                cl = 2 
            elif sur <= 300:
                cl = 0
            else: cl = 1
    except: pass 
    seg, t1ce, t1, t2, flair = [], [], [], [], []
    for name in names:
        name = os.path.join(line, name)
        
        if "_seg.nii" in name:
            seg.append(name)
        elif "_t1ce.nii" in name:
            t1ce.append(name)
        elif "_t1.nii" in name:
            t1.append(name)
        elif "_t2.nii" in name:
            t2.append(name)
        elif "_flair.nii" in name:
            flair.append(name)
    if stage == 'test':
#         print(f'{"image": t1ce + t1 + t2 + flair}')
        return {"image": t1ce + t1 + t2 + flair}
    else:
        try:
            if sur != None:
                if cl!= None:
                    return {"label": seg[0], "image": t1ce + t1 + t2 + flair, "survival": sur, "class": cl }
                else: 
                    return {"label": seg[0], "image": t1ce + t1 + t2 + flair, "survival": sur,}
        except: pass
    



def produce_datalist(dataset_dir, stage, split_lh, csv=None, clas =False):
    """
    This function is used to split the dataset.
    It will produce 200 samples for training, and the other samples are divided equally
    into val and test sets.
    """
#     print("split lh ", split_lh)
    if split_lh=="true":
        samples = sorted(glob.glob(os.path.join(dataset_dir, "*", "*"), recursive=True))
    else:
        samples = sorted(glob.glob(os.path.join(dataset_dir, "*/"), recursive=True))
    datalist = []
    for line in samples:
        if csv:
            df = pd.read_csv(csv, index_col = "BraTS18ID")
            t = produce_sample_dict_csv(line, stage, df, clas)
            if t != None:
                datalist.append(t)
        else:
            t = produce_sample_dict(line, stage)
            if t != None:
                datalist.append(t)
    if stage == 'train':
        train_list, val_list = train_test_split(datalist, train_size=0.7)
        return {"training": train_list, "validation": val_list, "testing": val_list}
    else:
        return {"test": datalist}


def main(args):
    """
    split the dataset and output the data list into a json file.
    """
#     os.system("mkdir /configs")
    data_file_base_dir = os.path.join(os.path.abspath(args.path))
    output_json = args.output
    # produce deterministic data splits
    monai.utils.set_determinism(seed=123)
    datalist = produce_datalist(data_file_base_dir, args.stage, args.split, args.csv, args.clas)
    with open(output_json, "w") as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path",
        type=str,
        default="/content/brats_2018",
        help="root path of brats 2018 dataset.",
    )
    parser.add_argument(
        "--output", type=str, default="configs/datalist.json", help="relative path of output datalist json file.",
    )
    parser.add_argument(
        "--stage", type=str, default='train', help=""
    )
    parser.add_argument(
        "--split", type=str, default="true",
    )
    parser.add_argument(
        "--csv", type = str, default=None
    )
    parser.add_argument(
        "--clas", type = bool, default= False
    )
    args = parser.parse_args()

    main(args)



# python libs/data/prepare_datalist.py --path temp/data_test/test --output temp/datalist.json --stage "train" --split "false" --csv temp/data_test/survival_data.csv
# python libs/data/prepare_datalist.py --path /kaggle/input/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Validation --stage "test" --split "false" --output /kaggle/working/datalist.json --csv /kaggle/input/miccai-brats2018-original-dataset/MICCAI_BraTS_2018_Data_Validation/survival_evaluation.csv