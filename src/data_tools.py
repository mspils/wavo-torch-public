"""
Provides tools to generate dataset for waterlevel forecasting.
The important part is the WaVoDataModule class
"""

import pickle

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


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
        scaler (StandardScaler, optional): Scaler to use for input data. Only given when the DataModule is created for prediction. Defaults to None.
        target_min (float, optional): Minimum value of target data. Only given when the DataModule is created for prediction. Defaults to None.
        target_max (float, optional): Maximum value of target data. Only given when the DataModule is created for prediction. Defaults to None.
    """

    def __init__(self, filename, level_name_org,
            in_size=144,
            out_size=48,
            batch_size=2048,
            differencing=0,
            percentile=0.95,
            train=0.7,
            val=.15,
            scaler=None,
            target_min=None,
            target_max=None,
            **kwargs) -> None:
        super().__init__()
        self.filename = str(filename)
        self.gauge_column = level_name_org
        self.target_column = level_name_org
        self.in_size = in_size
        self.out_size = out_size
        self.differencing = differencing
        self.percentile = percentile
        self.train = train
        self.val = val
        self.scaler = scaler
        self.target_min = torch.tensor(target_min, dtype=torch.float32) if isinstance(target_min,float) else target_min #this is kinda messy
        self.target_max = torch.tensor(target_max, dtype=torch.float32) if isinstance(target_max,float) else target_max
        self.threshold = None
        self.feature_count = None
        self.save_hyperparameters()

    def prepare_data(self):

        if (self.scaler is None and
            self.target_min is None and
            self.target_max is None and
            self.threshold is None and
            self.feature_count is None):

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
            train= df[:val_idx]

            self.feature_count = train.shape[-1]
            _, y_train_temp = self.create_dataset(train)


            self.target_min = y_train_temp.min()
            self.target_max = y_train_temp.max()
            self.scaler = StandardScaler()
            self.scaler.fit(train)


            self.save_hyperparameters({
                "target_min": self.target_min.item(),
                "target_max": self.target_max.item(),
                "scaler": pickle.dumps(self.scaler),
                "threshold": self.threshold})
        elif (self.scaler is None or
            self.target_min is None or
            self.target_max is None or
            self.threshold is None or
            self.feature_count is None):
            raise ValueError("If you provide a scaler, you also need to provide target_min, target_max, threshold and feature_count")

    def setup(self, stage: str):
        # TODO implement stages. skip loading if already loaded

        df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
        df = fill_missing_values(df)

        if self.differencing == 1:
            self.target_column = 'd1'
            df[self.target_column] = df[self.gauge_column].diff()

        # drop the first value, it's nan if differencing, but for comparing we need to always drop it
        df = df[1:]

        val_idx = int(self.train*len(df))
        test_idx = int(val_idx + self.val*len(df))

        #if stage == 'fit':
        train, val, test = df[:val_idx], df[val_idx:test_idx], df[test_idx:]


        # Normalize input data and append min max scaled targets.
        train_ss = pd.DataFrame(self.scaler.transform(train), index=train.index, columns=train.columns)
        train_ss['target'] = (train[self.target_column]- self.target_min.item()) / (self.target_max.item()-self.target_min.item())
        val_ss = pd.DataFrame(self.scaler.transform(val), index=val.index, columns=val.columns)
        val_ss['target'] = (val[self.target_column]- self.target_min.item()) / (self.target_max.item()-self.target_min.item())
        test_ss = pd.DataFrame(self.scaler.transform(test), index=test.index, columns=test.columns)
        test_ss['target'] = (test[self.target_column]- self.target_min.item()) / (self.target_max.item()-self.target_min.item())


        X_train, self.y_train = self.create_dataset(train_ss,target_idx=-1)
        X_val,  self.y_val = self.create_dataset(val_ss,target_idx=-1)
        X_test, self.y_test = self.create_dataset(test_ss,target_idx=-1)

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

    def create_dataset(self, df,target_idx=None):  # -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms a dataframe into a tuple of Tensors, where each element contains
        the last in_size input values/ next out_size target values

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            X,y: tuple of Tensors
        """
        dataset = df.values
        if target_idx is None:
            target_idx = df.columns.get_loc(self.target_column)

        X, y = [], []
        for i in range(len(dataset)-self.in_size-self.out_size):
            feature = dataset[i:i+self.in_size,:-1]
            target = dataset[i+self.in_size:i +
                             self.out_size+self.in_size, target_idx]
            X.append(feature)
            y.append(target)

        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_forecast_tensor(self,start,mode='single'):
        """if mode is single, returns a scaled tensor and a pd.DatetimeIndex
        if mode is ensemble, returns a dataframe with unscaled values and a pd.DatetimeIndex

        Basically this function loads data, does some preprocessing and cuts out the part that is needed for forecasting

        Args:
            start (str): start time of the forecast -1h
            mode (str, optional): single or ensemble. Defaults to 'single'.

        Returns:
            _type_: _description_
        """
        df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
        df = fill_missing_values(df)

        if self.differencing == 1:  # TODO actually implement differencing
            self.target_column = 'd1'
            df[self.target_column] = df[self.gauge_column].diff()

        if start is None:
            start_index = 0
        else:
            start_time = pd.to_datetime(start)
            start_index = max(0, df.index.get_indexer(
                [start_time], method='nearest')[0] - self.in_size + 1)

        forecast_index = pd.DatetimeIndex([start_time], freq='h')

        end_index = start_index + self.in_size

        df = df[start_index:end_index]

        if mode == 'ensemble':
            return df, forecast_index

        df_scaled = pd.DataFrame(self.scaler.transform(df), index=df.index, columns=df.columns)
        X = torch.tensor(np.array([df_scaled.values]), dtype=torch.float32)

        return X, forecast_index


    def get_data_forecast(self, start=None, end=None, mode='evaluate'):
        """_summary_

        Args:
            start (_type_, optional): _description_. Defaults to None.
            end (_type_, optional): _description_. Defaults to None.
        """
        df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
        df = fill_missing_values(df)

        if self.differencing == 1:  # TODO actually implement differencing
            self.target_column = 'd1'
            df[self.target_column] = df[self.gauge_column].diff()

        if start is None:
            start_index = 0
        else:
            start_time = pd.to_datetime(start)
            start_index = max(0, df.index.get_indexer(
                [start_time], method='nearest')[0] - self.in_size + 1)

        if mode == 'single':
            end_index = start_index + self.in_size
        elif end is None and mode == 'evaluate':
            end_index = len(df)
        else:
            end_time = pd.to_datetime(end)
            end_index = min(len(df), df.index.get_indexer(
                [end_time], method='nearest')[0] + self.out_size + 1)  # TODO check if this is correct

            # end is None and mode == 'single':

        df = df[start_index:end_index]
        target_idx = df.columns.get_loc(self.target_column)
        y_true = df.iloc[self.in_size-1:-(self.out_size+1), target_idx]

        df_scaled = pd.DataFrame(self.scaler.transform(
            df), index=df.index, columns=df.columns)

        if mode == 'single':
            X = torch.tensor(np.array([df_scaled.values]), dtype=torch.float32)
            y = torch.zeros((1, self.out_size), dtype=torch.float32)
            y_true = pd.DatetimeIndex(
                [start_time], freq='h')
            dataset = MyDataset(X, y)
            data_loader = DataLoader(dataset, 1, num_workers=1)

        else:
            df_scaled['target'] = (df[self.target_column]- self.target_min.item()) / (self.target_max.item()-self.target_min.item())

            X, y = self.create_dataset(df_scaled)
            dataset = MyDataset(X, y)
            data_loader = DataLoader(
                dataset, batch_size=self.hparams.batch_size, num_workers=8)

        return df, data_loader, y_true
