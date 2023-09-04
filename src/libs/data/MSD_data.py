import os
import tempfile
import monai
from monai import transforms
from monai import data
from monai.apps import DecathlonDataset

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print("--"*30)
print("DATA DIR: ",root_dir)
print("--"*30)

def get_loader(batch_size = 1, task='Task01_BrainTumour', root_dir=root_dir):

    train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.EnsureTyped(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys="image"),
            transforms.EnsureTyped(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys="image"),
            transforms.EnsureTyped(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    train_ds = DecathlonDataset(
        root_dir=root_dir,
        task=task,
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
    )
    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task=task,
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
    )
    train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=False,)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False,)

    return train_loader, val_loader


if __name__ == "__main__":
    get_loader()