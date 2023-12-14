import os.path

import torch
import time
from colorama import Fore
import numpy as np


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def log(str="", level="info"):
    assert level in ['info', 'warning', 'error', 'important', "Debug", "Output"]
    information = f"[" + level.upper() + "] " + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ': ' + str
    if level == "info":
        pass
    elif level == "warning":
        information = Fore.BLUE + information + Fore.RESET
    elif level == "error":
        information = Fore.RED + information + Fore.RESET
    elif level == "important":
        information = Fore.MAGENTA + information + Fore.RESET
    elif level == "Debug":
        information = Fore.BLUE + information + Fore.RESET
    elif level == "Output":
        information = Fore.GREEN + information + Fore.RESET
    print(information)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            log(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def parser_args(args):
    if args.root_path == "ETTh1":
        args.data_path = "ETTh1.csv"
        args.target = 7  # the index of "OT" in csv file
        args.ration = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        args.c_out = 7
        args.enc_in = 7
        args.dec_in = 7

    if args.root_path == "SML2010":
        args.pre_lens = (24, 48, 96, 192, 336, 480)
    else:
        args.pre_lens = (24, 48, 96, 192, 336, 480, 624, 720)

    return args
