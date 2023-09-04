from monai.networks.nets import DynUNet, SegResNet
import torch


def get_trained(dyn_trained, seg_trained):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dyn = DynUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            filters = [32, 64, 128, 256, 320, 320],
            kernel_size= [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides= [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[ [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=1).to(device)
    checkpoint = torch.load(dyn_trained ,map_location=torch.device(device))
    dyn.load_state_dict(checkpoint['model'])
    dyn = dyn.to(device)
    dyn = torch.nn.Sequential(dyn.skip_layers)

    seg = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,).to(device)

    checkpoint = torch.load(seg_trained ,map_location=torch.device(device))
    seg.load_state_dict(checkpoint['model'])
    seg = seg.to(device)
    seg = torch.nn.Sequential(seg.act_mod, seg.convInit, *seg.down_layers, *seg.up_samples)
    del checkpoint
    return dyn.eval(), seg.eval()

