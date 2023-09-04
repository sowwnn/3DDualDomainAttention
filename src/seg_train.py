import torch
from torch import nn
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete

import wandb
import json
import os
import time
import argparse


from model.st.get_baseline import get_model
from libs.data.dataset import dataloader
from libs.data.st_dataset import dataloader as st_dataloader
from libs.data.iseg_dataset import iseg_dataloader
from libs.data.mrbrains_dataset import mrbrains_dataloader


def train_step(model, train_loader, optimizer, loss_function, lr_scheduler, logger, config, scaler, epoch):
    step = 0
    epoch_loss = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            output = model(inputs)
            loss = loss_function(output, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        # print(
        #     f"{step}/{len(train_loader)}"
        #     f", train_loss: {loss.item():.4f}"
        #     f", step time: {(time.time() - step_start):.4f}"
        # )
        if step % 10 == 0:
            print("=",end="")
        if logger:
            logger.log({"loss":loss.item()})

    lr = optimizer.param_groups[0]['lr']
    if logger:
        logger.log({"lr":lr})
    epoch_loss /= step
    lr_scheduler.step()
    values['epoch_loss_values'].append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} lr: {lr:.4f}")


def valid_step(model, val_loader,optimizer, dice_metric, dice_metric_batch, logger, config, epoch):
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs =  sliding_window_inference(inputs=val_inputs,  roi_size=(128, 128, 128),  sw_batch_size=1,  predictor=model,  overlap=0.5,)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            values['metric_values'].append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            values['metric_values_tc'].append(metric_tc)
            metric_wt = metric_batch[1].item()
            values['metric_values_wt'].append(metric_wt)
            metric_et = metric_batch[2].item()
            values['metric_values_et'].append(metric_et)

            if logger:
                logger.log({
                    "metric_mean": metric,
                    "metric_wt": metric_wt,
                    "metric_tc": metric_tc,
                    "metric_et": metric_et,
                    "epoch": epoch,
                })


            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > values['best_metric']:
                values['best_metric'] = metric
                values['best_metric_epoch'] = epoch + 1
                values['best_metrics_epochs_and_time'][0].append(metric)
                values['best_metrics_epochs_and_time'][1].append(epoch + 1)
                values['best_metrics_epochs_and_time'][2].append(time.time() - total_start)
                torch.save(
                    {   'epoch': epoch +1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                    os.path.join(config["results_dir"], "best_metric_model.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {values['best_metric']:.4f}"
                f" at epoch: {values['best_metric_epoch']}")
            torch.save(
                    {   'epoch': epoch +1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                    os.path.join(config["results_dir"], "last_model.pth"),
                )
            print("saved metric model")
            
def inference(input, model):
    def _compute(input, model):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input, model)
    else:
        return _compute(input, model)
        
def run(model, train_loader, val_loader, optimizer, loss_function, lr_scheduler, metric, metric_batch, logger, config, sepoch = 0):
    os.system(f"mkdir {config['results_dir']}")
    ################ INIT #####################
    global values
    values = {
        'best_metric' : -1,
        'best_metric_epoch' : -1,
        'best_metrics_epochs_and_time' : [[], [], []],
        'epoch_loss_values' : [],
        'metric_values' : [],
        'metric_values_tc' : [],
        'metric_values_wt' : [],
        'metric_values_et' : [],
    }
    
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True
    
    global VAL_AMP
    global post_trans
    global total_start
    global device
    VAL_AMP = True
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_start = time.time()
    
    ################ RUN #####################
    
    for epoch in range(sepoch, config['max_epochs']):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config['max_epochs']}")
        model.train()
        epoch_loss = 0
        
        train_step(model, train_loader, optimizer, loss_function, lr_scheduler, logger, config, scaler, epoch)
    
        if (epoch + 1) % config["step_val"] == 0:
            valid_step(model, val_loader,optimizer, metric, metric_batch, logger, config, epoch)

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        total_time = time.time() - total_start
        with open(f"{config['results_dir']}/output.json", "w") as outfile:
            json.dump(values, outfile)
        
        with open(f"{config['results_dir']}/config.json", "w") as outfile:
            json.dump(config, outfile)


        if (time.time() - total_start) > 42000:
            break

def main(data):
    ## Init model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(data['model_name'], data['att'], data['in_channel'],  data['out_channel'])
    model = model.to(device)


    ##Init dataloader
    datalist = data['datalist']
    with open(datalist) as f:
        datalist = json.load(f)
    if data["project"] == "survival_time":
        train_loader, val_loader = st_dataloader(datalist, 1, 'train', True) 
    if data['project'] == "iseg":
        train_loader, val_loader = iseg_dataloader(datalist,1 ,'train', True)
    if data['project'] == "mrbrains":
        train_loader, val_loader = mrbrains_dataloader(datalist,1 ,'train', True)
    else:
        train_loader, val_loader = dataloader(datalist, 1, 'train', True) 


    ##Init method and params
    config = data['config']
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=5e-1, squared_pred=True, to_onehot_y=False, sigmoid=True)
    metric = DiceMetric(include_background=True, reduction="mean")
    metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['tmax'], eta_min=1e-5)

    if data['model_trained']:
        checkpoint = torch.load(data['model_trained'],map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        if data["reset_epoch"]:
            epoch = 0
        else:
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        epoch = 0

    logger = None
    if config['log'] == True:
        logger = wandb.init(project=data['project'], name = config['name'], config=config, dir="/kaggle/input/pretrain/BrainTumour_Seg/")

    ## Run
    run(model, train_loader, val_loader, optimizer, loss_function, lr_scheduler, metric, metric_batch, logger, config, sepoch = epoch)

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input",
        type=str,
        default="/content/exp.json",
        help="expriment configuration",
    )
    args = parser.parse_args()
    f = open(args.input)
    data = json.load(f)

    main(data)
    
            
    