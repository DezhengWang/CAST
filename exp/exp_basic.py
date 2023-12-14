from tools import EarlyStopping
from eval.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data.data_loader import load_dataloader
import os
import time
from tools import try_gpu, log
from thop import profile

import warnings

warnings.filterwarnings('ignore')


class Exp_Basic():
    def __init__(self, args):
        super(Exp_Basic, self).__init__()
        self.args = args
        self.cuda = args.cuda
        self.net = args.net
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if isinstance(self.cuda, int):
            device = try_gpu(self.cuda)
            if device.type == "cuda":
                os.environ["CUDA DEVICES"] = str(self.cuda)
                log('Use GPU: CUDA:{}'.format(self.cuda), level="important")
            else:
                counts = torch.cuda.device_count()
                device = torch.device('cpu')
                log('Use CPU, Not enough GPU devices, {} in total, expected {}'.format(counts, self.cuda + 1),
                    level="important")
        elif isinstance(self.cuda, list):
            device = try_gpu(self.cuda[-1])
            if device.type == "cuda":
                os.environ["CUDA DEVICES"] = str(",".join(self.cuda))
                log('Use GPU: CUDA:{}'.format(",".join(self.cuda)), level="important")
            else:
                counts = torch.cuda.device_count()
                device = torch.device('cpu')
                log('Use CPU, Not enough GPU devices, {} in total, expected {}'.format(counts, self.cuda[-1] + 1),
                    level="important")
        else:
            device = torch.device('cpu')
            log('Use CPU', level="important")
        return device

    def _build_model(self):
        net = self.net(self.args).float()
        net = net.to(self.device)
        return net

    def _get_data(self):
        train_loader, valid_loader, test_loader = load_dataloader(self.args)
        return train_loader, valid_loader, test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train_loop(self, train_loader, valid_loader):

        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=self.args.step_size, gamma=0.5)
        criterion = self._select_criterion()

        losses = []
        times = []
        best_valid = 1000
        for epoch in range(1, self.args.epochs + 1):
            train_loss, during_time = self.train(train_loader, optim, criterion)
            valid_res = self.valid(valid_loader)
            early_stopping(valid_res["MSE"], self.model, path)

            if valid_res["MSE"] < best_valid:
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
                best_valid = valid_res["MSE"]

            # log loss
            losses.append([train_loss, valid_res["MSE"]])
            times.append(during_time)
            log(
                f'{self.net.__name__} on {self.args.root_path}, Epoch: {epoch}/{self.args.epochs}' + ' Train: {:.4f}, Eval: {:.4f}, lr: {}'.format(
                    train_loss, valid_res["MSE"], scheduler.get_lr()))

            if early_stopping.early_stop:
                log("Early stopping", level="important")
                break
            scheduler.step(epoch)

        self.train_time = np.mean(np.array(times))

        best_model_path = os.path.join(self.args.checkpoints, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def train(self, train_loader, optim, criterion):
        self.model.train()
        train_loss = []
        start_time = time.time()

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optim.zero_grad()

            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

            loss = criterion(pred, true)
            train_loss.append(loss.item())

            loss.backward()
            optim.step()

        during_time = time.time() - start_time
        return np.average(train_loss), during_time

    def valid(self, valid_loader):
        self.model.eval()
        Y_hat = None
        Y = None
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            y_hat, y = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            if Y_hat is None:
                Y_hat = y_hat.detach().cpu().numpy()
                Y = y.detach().cpu().numpy()
            else:
                Y_hat = np.vstack((Y_hat, y_hat.detach().cpu().numpy()))
                Y = np.vstack((Y, y.detach().cpu().numpy()))

        mae, mse, rmse, mape, mspe = metric(Y_hat, Y)

        return {"MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "MSPE": mspe}

    def test(self, test_loader, load=False):
        if load:
            best_model_path = os.path.join(self.args.checkpoints, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        flops, params = None, None
        Y_hat = None
        Y = None
        times = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                start_time = time.time()
                y_hat, y = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                during_time = time.time() - start_time
                times.append(during_time)

                if Y_hat is None:
                    Y_hat = y_hat.detach().cpu().numpy()
                    Y = y.detach().cpu().numpy()
                else:
                    Y_hat = np.vstack((Y_hat, y_hat.detach().cpu().numpy()))
                    Y = np.vstack((Y, y.detach().cpu().numpy()))

            ## Calculate Params and FLOPs
            batch_x = batch_x[0:1].float().to(self.device)
            batch_y = batch_y[0:1].float()
            batch_x_mark = batch_x_mark[0:1].float().to(self.device)
            batch_y_mark = batch_y_mark[0:1].float().to(self.device)

            dec_inp = torch.zeros([batch_y.shape[0], self.args.iter_horizon, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.size[1], :], dec_inp], dim=1).float().to(self.device)

            flops, params = profile(self.model, inputs=(
                batch_x, batch_x_mark, dec_inp, batch_y_mark[:, :self.args.size[1] + self.args.iter_horizon]),
                                    verbose=False)

        inference_time = np.mean(np.array(times))
        mae, mse, rmse, mape, mspe = metric(Y_hat, Y)
        log("{} on {}, FLOPs: {:.2}G, Params: {:.2}M, MSE: {:.4}\n".format(self.net.__name__,
                                                                           self.args.root_path,
                                                                           flops / 1e9,
                                                                           params / 1e6,
                                                                           mse), level="Output")
        # result save
        np.savez_compressed(os.path.join(self.args.checkpoints, 'results'),
                            Y=Y,
                            Y_hat=Y_hat,
                            nParams=params,
                            FLOPs=flops,
                            InferenceTime=inference_time,
                            TrainTime=self.train_time
                            )

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], self.args.size[2], batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:, :self.args.size[1], :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.features == 'S' or self.args.features == 'MS':
            batch_y = batch_y[:, -self.args.size[2]:, self.args.target - 1:self.args.target].to(self.device)
        else:
            batch_y = batch_y[:, -self.args.size[2]:, :].to(self.device)

        return outputs, batch_y

