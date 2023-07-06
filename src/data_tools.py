"""
Provides tools to generate dataset for waterlevel forecasting.
The important part is the WaVoDataModule class
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler  # , MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
import numpy as np
import torch
import pickle


def fill_missing_values(df: pd.DataFrame, max_fill=10):
    """Fills values in a DataFrame.
    First columns recognized as Precipitation columns (containing NEW or NVh) are filled with 0,
    other columns are filled linear with a limit of 24, with warning if more than 5 continuos values are missing

    Args:
        df : DataFrame that might be missing values
        max_fill : How many continuosly missing values are allowed
    Returns:
        DataFrame: Filled DataFrame (if possible)
    """
    old_size = df.shape[0]

    df = df.resample('h').mean()
    na_count = df.isna().sum(axis=0)

    # get all columns with precipitation and fill missing values with 0
    mask = df.columns.str.contains('NEW') | df.columns.str.contains('NVh')

    prec_cols = list(na_count[mask][na_count[mask] > 0].index)
    if len(prec_cols) > 0:
        df.loc[:, mask] = df.loc[:, mask].fillna(0)

    # interpolate data in all other columns
    df = df.interpolate(limit=max_fill, limit_direction='both')

    if df.isna().sum().sum() > 0:
        raise LargeGapError(f"Some columns were missing more than {max_fill} continuous values, either raise the limit or fill values manually." +
                            f"{df.isna().sum().sum()} still missing, maybe due to {len(df)-old_size} missing timestamps?")
    return df


class LargeGapError(Exception):
    """Basic Custom error, thrown if a gap in a dataset is too large

    Args:
        Exception (Exception):
    """


class MyDataset(Dataset):
    """Custom trivial Dataset, because using the normal one doesn't work for unknown reasons
    Args:
        X (Tensor): Input values 
        y (Tensor): Target values
    """

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # TODO differencing


class WaVoDataModule(pl.LightningDataModule):
    """Provides dataloades for waterlevel forecasting
    Args:
        filename (str|Path): Path to .csv file, first column should be a timeindex
        level_name_org (str): Name of the column with target values
        in_size (int, optional): input length in hours. Defaults to 144.
        out_size (int, optional): output length in hours. Defaults to 48.
        batch_size (int, optional): batch_size. Defaults to 128.
        differencing (int, optional): How offen to difference the timeseries (not yet implemented). Defaults to 0.
        percentile (float, optional): percentile to define flood values. If > 1 it is used as an absolute value, not percentile. Defaults to 0.95.
        train (float, optional): share of training data (first train %). Defaults to 0.7.
        val (float, optional): share of validation data (directly after train). Defaults to .15.
        test (float, optional):share of test data (directly after val) DOESN'T ACTUALLY DO ANYTHING. Defaults to .15.   
    """

    # pylint: disable-next=unused-argument
    def __init__(self, filename, level_name_org, in_size=144, out_size=48, batch_size=2048, differencing=0, percentile=0.95, train=0.7, val=.15, test=.15, start=None, scaler=None, mean=None, scale=None, **kwargs) -> None:
        super().__init__()
        self.filename = filename
        # self.batch_size = batch_size
        self.gauge_column = level_name_org
        self.target_column = level_name_org
        self.in_size = in_size
        self.out_size = out_size
        self.differencing = differencing
        self.percentile = percentile
        self.train = train
        self.val = val
        self.start = start
        self.scaler = scaler
        self.mean = mean
        self.scale = scale
        self.save_hyperparameters(ignore='start')  # TODO ignore?

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # TODO implement stages

        df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
        df = fill_missing_values(df)

        if self.differencing == 1:
            self.target_column = 'd1'
            df[self.target_column] = df[self.gauge_column].diff()

        # drop the first value, it's nan if differencing, but for comparing we need to always drop it
        df = df[1:]
        if 0 < self.percentile < 1:
            self.threshold = df[self.gauge_column].quantile(
                self.percentile).item()
        else:
            self.threshold = self.percentile

        val_idx = int(self.train*len(df))
        test_idx = int(val_idx + self.val*len(df))
        train, val, test = df[:val_idx], df[val_idx:test_idx], df[test_idx:]

        if self.mean is None and self.scale is None and self.scaler is None:
            _, y_train_temp = self.create_dataset(train)

            ss_target = StandardScaler()
            ss_target.fit(y_train_temp)

            self.mean = ss_target.mean_.mean().item()
            # TODO use actual scalers?
            # TODO save scaler somehow
            self.scale = ss_target.scale_.mean().item()

            self.scaler = StandardScaler()
            self.scaler.fit(train)

            self.save_hyperparameters({"mean": self.mean,
                                       "scale": self.scale,
                                       "scaler": pickle.dumps(self.scaler),
                                       "threshold": self.threshold})
        else:
            assert self.mean is not None and self.scale is not None and self.scaler is not None

        train_ss = pd.DataFrame(self.scaler.transform(
            train), index=train.index, columns=train.columns)
        val_ss = pd.DataFrame(self.scaler.transform(
            val), index=val.index, columns=val.columns)
        test_ss = pd.DataFrame(self.scaler.transform(
            test), index=test.index, columns=test.columns)

        X_train, self.y_train = self.create_dataset(train_ss)
        X_val,  self.y_val = self.create_dataset(val_ss)
        X_test, self.y_test = self.create_dataset(test_ss)
        self.feature_count = X_train.shape[-1]

        self.train_set = MyDataset(X_train, self.y_train)
        self.val_set = MyDataset(X_val, self.y_val)
        self.test_set = MyDataset(X_test, self.y_test)

        # self.save_hyperparameters({"mean": self.mean, "scale": self.scale})

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        # return the trainset, but sorted
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)

    def create_dataset(self, df):
        """Transforms a dataframe into a tuple of Tensors, where each element contains 
        the last in_size input values/ next out_size target values

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        dataset = df.values

        target_idx = df.columns.get_loc(self.target_column)

        X, y = [], []
        for i in range(len(dataset)-self.in_size-self.out_size):
            feature = dataset[i:i+self.in_size]
            target = dataset[i+self.in_size:i +
                             self.out_size+self.in_size, target_idx]
            X.append(feature)
            y.append(target)

        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_data_forecast(self, start=None, end=None):
        """_summary_

        Args:
            start (_type_, optional): _description_. Defaults to None.
            end (_type_, optional): _description_. Defaults to None.
        """
        df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
        df = fill_missing_values(df)

        if self.differencing == 1:
            self.target_column = 'd1'
            df[self.target_column] = df[self.gauge_column].diff()

        if start is None:
            start = 0

        if end is None:
            end = len(df)
        df = df[start:end]

        # TODO start_index function
        # _, y_org = self.create_dataset(df)
        target_idx = df.columns.get_loc(self.target_column)
        y_true = df.iloc[self.in_size-1:-(self.out_size+1), target_idx]

        df_scaled = pd.DataFrame(self.scaler.transform(
            df), index=df.index, columns=df.columns)
        X, y = self.create_dataset(df_scaled)

        dataset = MyDataset(X, y)
        data_loader = DataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=8)

        return df, data_loader, y_true


class WaVoDataModuleCLI(pl.LightningDataModule):
    """Provides dataloades for waterlevel forecasting
    Args:
        filename (str|Path): Path to .csv file, first column should be a timeindex
        level_name_org (str): Name of the column with target values
        in_size (int, optional): input length in hours. Defaults to 144.
        out_size (int, optional): output length in hours. Defaults to 48.
        batch_size (int, optional): batch_size. Defaults to 128.
        differencing (int, optional): How offen to difference the timeseries (not yet implemented). Defaults to 0.
        percentile (float, optional): percentile to define flood values. If > 1 it is used as an absolute value, not percentile. Defaults to 0.95.
        train (float, optional): share of training data (first train %). Defaults to 0.7.
        val (float, optional): share of validation data (directly after train). Defaults to .15.
        test (float, optional):share of test data (directly after val) DOESN'T ACTUALLY DO ANYTHING. Defaults to .15.   
    """

    # pylint: disable-next=unused-argument
    def __init__(self, filename, level_name_org, in_size=144, out_size=48, batch_size=2048, differencing=0, percentile=0.95, train=0.7, val=.15, test=.15, **kwargs) -> None:
        super().__init__()
        self.filename = filename
        # self.batch_size = batch_size
        self.gauge_column = level_name_org
        self.target_column = level_name_org
        self.in_size = in_size
        self.out_size = out_size
        self.differencing = differencing
        self.percentile = percentile
        self.train = train
        self.val = val

        # self.save_hyperparameters()

        # This is ugly, but necessary for the CLI to work
        df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
        df = fill_missing_values(df)

        if self.differencing == 1:
            self.target_column = 'd1'
            df[self.target_column] = df[self.gauge_column].diff()

        # drop the first value, it's nan if differencing, but for comparing we need to always drop it
        df = df[1:]
        if 0 < self.percentile < 1:
            self.threshold = df[self.gauge_column].quantile(
                self.percentile).item()
        else:
            self.threshold = self.percentile

        val_idx = int(self.train*len(df))
        df_train = df[:val_idx]

        X_train_temp, y_train_temp = self.create_dataset(df_train)
        self.feature_count = X_train_temp.shape[-1]

        ss_target = StandardScaler()
        ss_target.fit(y_train_temp)
        self.mean = ss_target.mean_.mean().item()  # TODO use actual scalers?
        self.scale = ss_target.scale_.mean().item()

        # del train

        self.save_hyperparameters()

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
        df = fill_missing_values(df)

        if self.differencing == 1:
            self.target_column = 'd1'
            df[self.target_column] = df[self.gauge_column].diff()

        # drop the first value, it's nan if differencing, but for comparing we need to always drop it
        df = df[1:]
        if 0 < self.percentile < 1:
            self.threshold = df[self.gauge_column].quantile(
                self.percentile).item()
        else:
            self.threshold = self.percentile

        val_idx = int(self.train*len(df))
        test_idx = int(val_idx + self.val*len(df))
        train, val, test = df[:val_idx], df[val_idx:test_idx], df[test_idx:]

        # _, y_train_temp = self.create_dataset(train)

        # ss_target = StandardScaler()
        # ss_target.fit(y_train_temp)
        # self.mean = ss_target.mean_.mean().item()  # TODO use actual scalers?
        # self.scale = ss_target.scale_.mean().item()

        ss = StandardScaler()
        train_ss = pd.DataFrame(ss.fit_transform(
            train), index=train.index, columns=train.columns)
        val_ss = pd.DataFrame(ss.transform(
            val), index=val.index, columns=val.columns)
        test_ss = pd.DataFrame(ss.transform(
            test), index=test.index, columns=test.columns)

        X_train, self.y_train = self.create_dataset(train_ss)
        X_val,  self.y_val = self.create_dataset(val_ss)
        X_test, self.y_test = self.create_dataset(test_ss)
        # self.feature_count = X_train.shape[-1]

        self.train_set = MyDataset(X_train, self.y_train)
        self.val_set = MyDataset(X_val, self.y_val)
        self.test_set = MyDataset(X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        # return the trainset, but sorted
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)

    def create_dataset(self, df):
        """Transforms a dataframe into a tuple of Tensors, where each element contains 
        the last in_size input values/ next out_size target values

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        dataset = df.values

        target_idx = df.columns.get_loc(self.target_column)

        X, y = [], []
        for i in range(len(dataset)-self.in_size-self.out_size):
            feature = dataset[i:i+self.in_size]
            target = dataset[i+self.in_size:i +
                             self.out_size+self.in_size, target_idx]
            X.append(feature)
            y.append(target)

        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
