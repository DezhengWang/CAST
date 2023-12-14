import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .tools import *
from data.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset2Timeseries(Dataset):
    def __init__(self,
                 root_path,
                 flag='train',
                 size=None,
                 ration=[12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
                 features='MS',
                 data_path='ETTh1.csv',
                 target='OT',
                 scale=True,
                 inverse=False,
                 timeenc=0,
                 freq='h',
                 aug=None,
                 maskrate=0.5):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.ration = ration
        self.aug = aug
        self.maskrate = maskrate
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join("./data/", self.root_path, self.data_path))

        border1s = [0, self.ration[0], self.ration[1]]
        border2s = [self.ration[0], self.ration[1], self.ration[2]]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            target = df_raw.columns[self.target]
            df_data = df_raw[[target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_x = np.nan_to_num(self.data_x)
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_y = np.nan_to_num(self.data_y)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.aug:
            index = np.random.choice(len(self.aug))
            aug = self.aug[index]
            seq_x = aug(seq_x)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def load_dataloader(args):
    root_path = args.root_path
    size = args.size
    ration = args.ration
    features = args.features
    data_path = args.data_path
    target = args.target
    scale = args.scale
    inverse = args.inverse
    aug = args.aug
    maskrate = args.maskrate
    batch_size = args.batch_size

    train_data = Dataset2Timeseries(root_path=root_path,
                                    flag='train',
                                    size=size,
                                    ration=ration,
                                    features=features,
                                    data_path=data_path,
                                    target=target,
                                    scale=scale,
                                    inverse=inverse,
                                    aug=aug,
                                    maskrate=maskrate)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    valid_data = Dataset2Timeseries(root_path=root_path,
                                    flag='valid',
                                    size=size,
                                    ration=ration,
                                    features=features,
                                    data_path=data_path,
                                    target=target,
                                    scale=scale,
                                    inverse=inverse)

    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              drop_last=True)

    test_data = Dataset2Timeseries(root_path=root_path,
                                   flag='test',
                                   size=size,
                                   ration=ration,
                                   features=features,
                                   data_path=data_path,
                                   target=target,
                                   scale=scale,
                                   inverse=inverse)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    return train_loader, valid_loader, test_loader
