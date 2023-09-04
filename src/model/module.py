import torch 
import numpy as np
import pytorch_lightning as pl
from monai.data import decollate_batch
from functools import partial
from monai.inferers import sliding_window_inference


class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, metric):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        
        self.metric = metric
        self.save_hyperparameters(ignore=['net','criterion'])
        self.scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        self.automatic_optimization = False
        self.metric_tc, self.metric_wt, self.metric_et = self.reset_metric()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
        

    def prepare_batch(self, batch):
        return batch['image'].to(self.device), batch['label'].to(self.device)

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        y_hat = torch.nn.Sigmoid()(y_hat)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        with torch.cuda.amp.autocast():
              y_hat, y = self.infer_batch(batch)
              loss = self.criterion(y_hat, y)
        self.manual_backward(loss)
        opt.step()
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        metric_tc, metric_wt, metric_et = self.metric(y_hat,y).mean(axis=0)
        self.update_metric(metric_tc, metric_wt, metric_et)
        self.log('val_loss', loss)
        return loss

    def reset_metric(self):
        return np.array([]),np.array([]),np.array([])
    
    def update_metric(self, metric_tc, metric_wt, metric_et):
        self.metric_tc = np.append(self.metric_tc,metric_tc)
        self.metric_wt = np.append(self.metric_wt,metric_wt)
        self.metric_et = np.append(self.metric_et,metric_et)

    def validation_epoch_end(self, outputs):
        self.log('metric_tc',self.metric_tc.mean())
        self.log('metric_wt',self.metric_wt.mean())
        self.log('metric_et',self.metric_et.mean())
        self.metric_tc, self.metric_wt, self.metric_et = self.reset_metric()

