import os
import SimpleITK as sitk
join = os.path.join

from sklearn.model_selection import train_test_split
import monai 
import json
monai.utils.set_determinism(seed=123)

#%% training set: convert hdr to nii
def conver_hdr2nii(hdrpath, niipath, labelpath, val_niipath, val_hdrpath):

    os.system(f"mkdir {niipath} {labelpath} {val_niipath}")
    train, test = [], []

    for i in range(1,11):
        label_name = f'subject-{i}-label.hdr'
        t1_name = f'subject-{i}-T1.hdr'
        t2_name = f'subject-{i}-T2.hdr'
        t1_img = sitk.ReadImage(f"{hdrpath}/{t1_name}", imageIO='NiftiImageIO')
        t2_img = sitk.ReadImage(f"{hdrpath}/{t2_name}")
        label_img = sitk.ReadImage(join(hdrpath, label_name))
        t1_savename = 'iseg_' + str(i) + '_0000.nii.gz'
        t2_savename = 'iseg_' + str(i) + '_0001.nii.gz'
        label_savename = 'iseg_' + str(i) + '.nii.gz'
        sitk.WriteImage(t1_img, join(niipath, t1_savename))
        sitk.WriteImage(t2_img, join(niipath, t2_savename))
        sitk.WriteImage(label_img, join(labelpath, label_savename))
        train.append({'label':[join(labelpath, label_savename)], 'image':[join(niipath, t1_savename), join(niipath, t2_savename)]})

    #%% validation set: convert hdr to nii
    val_hdrpath = "/kaggle/input/iseg19/iSeg-2019-Validation"
    val_niipath = "/kaggle/working/imagesTs"

    for i in range(11, 24):
        t1_name = 'subject-'+str(i)+'-T1.hdr'
        t2_name = 'subject-'+str(i)+'-T2.hdr'
        t1_img = sitk.ReadImage(f"{val_hdrpath}/{t1_name}")
        t2_img = sitk.ReadImage(f"{val_hdrpath}/{t2_name}")

        t1_savename = 'iseg_' + str(i) + '_0000.nii.gz'
        t2_savename = 'iseg_' + str(i) + '_0001.nii.gz'

        sitk.WriteImage(t1_img, join(val_niipath, t1_savename))
        sitk.WriteImage(t2_img, join(val_niipath, t2_savename))

        test.append({'image':[join(val_niipath, t1_savename), join(val_niipath, t2_savename)]})

    train_list, val_list = train_test_split(train, train_size=0.8)
    datalist = {"training": train, "validation": val_list, "testing": train}

    with open('/kaggle/working/datalist.json', "w") as f:
        json.dump(datalist, f)
