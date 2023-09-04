import os
import sys


print("##"*30)
print("Install libraries")
os.system("pip install -q SimpleITK")
os.system("pip install -q pytorch_lightning")
os.system("pip install -q monai")
os.system("pip install -q wandb")
os.system("pip install -q torch-summary")
os.system("pip install -q einops")

print("##"*30)
print("Setup Datafolder")
# os.system("mkdir /content/brats_2018/ /content/brats_2018/training")
# os.system("unzip /content/drive/MyDrive/mah_ws/dataset/BraTS2018/MICCAI_BraTS_2018_Data_Training.zip -d /content/brats_2018/training")
print("##"*30)
print("Login wandb")
if len(sys.argv) > 1:
    os.system(f"wandb login {sys.argv[1]}")
else:
    print("Not logged in to wandb yet!")
# cd /content/drive/MyDrive/mah_ws/ai_prj/3D_Tumor/public
