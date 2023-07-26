"""
Inference Script for WaVo Torch Version
"""
import argparse
import configparser
import logging
import pickle
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml

from data_tools import WaVoDataModule
from ensemble_dict import ens_dict  # TODO expand, maybe move to config file?
from models import WaVoLightningModule

use_cuda = torch.cuda.is_available()

if use_cuda:
    accelerator = 'gpu'
    torch.set_float32_matmul_precision('high')
else:
    accelerator = 'cpu'




def get_pred(model:pl.LightningModule, data_loader:torch.utils.data.DataLoader, y_true_or_index, mode='evaluate') -> pd.DataFrame:
    """Makes a prediction using a dataloader and returns a dataframe with the predictions.

    Args:
        model (pl.LightningModule): Model
        data_loader (torch.utils.data.DataLoader): dataloader
        y_true_or_index (_type_): Either the true values in a dataframe or the index of the dataframe if mode is 'single' or 'ensemble'
        mode (str, optional): model. Defaults to 'evaluate'.

    Returns:
        pd.DataFrame: prediction(s)
    """
    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False)
    pred = trainer.predict(model, data_loader)
    pred = np.concatenate(pred)

    if mode == 'evaluate':
        y_pred = np.concatenate(
            [np.expand_dims(y_true_or_index, axis=1), pred], axis=1)
        y_pred = pd.DataFrame(
            y_pred, index=y_true_or_index.index, columns=range(49))
    else:
        y_pred = pd.DataFrame(pred, index=y_true_or_index,
                              columns=range(1, 49))
    return y_pred


def parse_args() -> argparse.Namespace:
    """ Parse all the arguments and provides some help in the command line
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Make a prediction with a trained torch model. Last argument should be the mode.')

    subparsers = parser.add_subparsers(title='subcommands/modes',description='Choose any of the following modes:',help='sub-command help',dest='mode')

    parser.add_argument('--config', metavar='config_file', type=Path,help='The path to your config file. Command line parameters will take precedence.')
    parser.add_argument('--data', metavar='data_file', type=Path,help='The path to your input data.')
    parser.add_argument('--model',metavar='model_folder',type=Path, help='The path to your model folder')
    parser.add_argument('--target',metavar='target_file',type=Path,help='The path to your prediction file. (Or folder is mode is ensemble)')
    parser.add_argument('--log',metavar='log_file', type=Path,help='The path to your log file.')
    parser.add_argument('--max_fill',metavar='max_fill', type=int,help='Maximum size of gaps in the data to fill/interpolate')



    parser_evaluate = subparsers.add_parser('evaluate', help='Generates a csv file with true and predicted values from "--start" to "--end"')
    parser_evaluate.add_argument('--start', type=str,help='The timestamp 1h before the first predicted value (needs previous values including the one at start) e.g. 2021-12-15 18')
    parser_evaluate.add_argument('--end', type=str,help='A timestamp for the prediction end, if None predict until the end of the dataset, ')

    parser_single = subparsers.add_parser('single', help='Generates a zrx file with predictions for a single timestamp, needs "--start" and "--gauge_name"')
    parser_single.add_argument('--start', type=str,help='The timestamp 1h before the first predicted value')
    parser_single.add_argument('--gauge_name', type=str,help='The name of the gauge in the zrx file.')

    parser_ensemble = subparsers.add_parser('ensemble', help='Generates a zrx file with predictions for a single timestamp for each member of the ensemble, needs "--start","--gauge_name","--ensemble_folder" and "--input_backup_folder"')
    parser_ensemble.add_argument('--start', type=str,help='The timestamp 1h before the first predicted value (Should be the timestamp of the ICON forecast)')
    parser_ensemble.add_argument('--gauge_name', type=str,help='The name of the gauge in the zrx file.')
    parser_ensemble.add_argument('--ensemble_folder',type=Path,help='The path to your ensemble input data.')
    parser_ensemble.add_argument('--input_backup_folder',type=Path,help='Where to save the ensemble input data to.')




    return parser.parse_args()

def load_settings_model(model_dir: Path) -> dict:
    """LightingModules have a hparams attribute that contains all the hyperparameters used for training. This function loads them from the model folder.

    Args:
        model_dir (Path): Path to the model folder

    Returns:
        dict: The hyperparameters and values used for scaling etc.
    """
    with open(model_dir / 'hparams.yaml', 'r',encoding='utf-8') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        yaml_data['scaler'] = pickle.loads(yaml_data['scaler'])

    return yaml_data

def get_config(parsed_args:dict) -> dict:
    """Creates a dict with values from a config file and the command-line parameters

    Args:
        parsed_args (dict): Commandline parameters

    Returns:
        dict: A dict with all parameters needed for predictions.
    """
    if parsed_args.config is None:
        return parsed_args

    config_parser = configparser.ConfigParser()
    config_parser.read(parsed_args.config)


    config = {
        'mode':             parsed_args.mode,
        'filename' :        parsed_args.data or Path(config_parser['SETTINGS'].get('data_file')),
        'model_folder':      parsed_args.model or Path(config_parser['SETTINGS'].get('model_folder')),
        'target_file':       parsed_args.target or Path(config_parser['SETTINGS'].get('target_file')),
        'log_file':         parsed_args.log or Path(config_parser['SETTINGS'].get('log_file','predict_torch.log')),
        'max_fill':         parsed_args.max_fill or config_parser['SETTINGS'].getint('max_fill',24),
        'start':            parsed_args.start or config_parser['SETTINGS'].get('start'), #TODO maybe start from zero or use newest possible values depending on mode?
    }


    if parsed_args.mode == 'evaluate':
        config['end'] = parsed_args.end or config_parser['SETTINGS'].get('end',None)

    if parsed_args.mode == 'single' or parsed_args.mode == 'ensemble':
        config['gauge_name'] = parsed_args.gauge_name or config_parser['SETTINGS'].get('gauge_name')

    if parsed_args.mode == 'ensemble':
        config['ensemble_folder']   = parsed_args.ensemble_folder or Path(config_parser['SETTINGS'].get('ensemble_folder'))
        config['input_backup_folder'] = parsed_args.input_backup_folder or Path(config_parser['SETTINGS'].get('input_backup_folder'))

    config_model = load_settings_model(config['model_folder'])
    config_model.update(config)
    #TODO add way to overwrite target column

    # Create directories for logs and predictions if they don't exist
    create_dirs(config)


    return config_model

def create_dirs(config:dict):
    """Creates directories for logs and predictions if they don't exist

    Args:
        config (dict): Config dict
    """
    if not config['log_file'].parent.exists():
        config['log_file'].parent.mkdir(parents=True)
    if not config['target_file'].parent.exists():
        config['target_file'].parent.mkdir(parents=True)
    if 'input_backup_folder' in config and not config['input_backup_folder'].exists():
        config['input_backup_folder'].mkdir(parents=True)

class WaVoPredictor():
    """Class for making predictions with a trained model.
    Pretty much just a wrapper for the LightningModule and DataModule, with some helper functions.
    """

    def __init__(self,config,cuda=False) -> None:
        self.cuda = cuda
        self.config = config
        map_location = None if cuda else torch.device("cpu")

        self.model = WaVoLightningModule.load_from_checkpoint(next((config['model_folder'] / 'checkpoints').iterdir()),map_location=map_location)
        self.model.eval()
        self.data_module = WaVoDataModule(**config)
        #TODO better don't keep config, instead keep the 2-3 values actually needed.
        #TODO maybe refactor so that the df is not loaded each time a prediction is made using this object

    def evaluate(self,start=None,end=None,save=True)->pd.DataFrame:
        """Makes and saves a prediction from start to end using the LightningModules predict_step and a dataloader.
        Start and end are optional and will be taken from the config if not provided.
        Args:
            start (str, optional): The timestamp 1h before the first predicted value. Defaults to None.
            end (str, optional): A timestamp for the prediction end, if None predict until the end of the dataset. Defaults to None.
            save (bool, optional): Whether to save the generated prediction. Defaults to True.
        """
        #TODO test dropout/lstm memory here as well
        cur_start = start or self.config['start']
        cur_end = end or self.config['end']

        _, data_loader, y_true_or_index = self.data_module.get_data_forecast(start=cur_start,end=cur_end,mode='evaluate')
        y_pred = get_pred(self.model, data_loader, y_true_or_index, mode='evaluate')
        if save:
            logging.info("Saving csv prediction to %s", self.config['target_file'].resolve())
            y_pred.to_csv(self.config['target_file'])

        return y_pred

    def single(self,start=None,save=True)->pd.DataFrame:
        """Creates a prediction for a single timestamp and saves it to a zrx file (if save) and returns the dataframe.
        Start is optional and will be taken from the config if not provided.

        Args:
            start (str, optional): The timestamp 1h before the first predicted value. Defaults to None.
            save (bool, optional): Whether to save the generated prediction. Defaults to True.

        Returns:
            pd.DataFrame: zrx version of a prediction
        """
        cur_start = start or self.config['start']
        x, forecast_index = self.data_module.get_forecast_tensor(cur_start)
        # The timestamp 1h before the first predicted value. Defaults to None.

        # If we use cuda we need to move the tensor to the gpu and the predictions back to the cpu
        y_pred = self.get_pred_manually(x)
        y_pred = pd.DataFrame(y_pred, index=forecast_index,columns=range(1, 49))
        df_zrx = self._to_zrx(y_pred,member=1)
        if save:
            zrx_name = self.config['target_file']
            self._save_zrx(df_zrx,zrx_name)

        return df_zrx

    def ensemble(self,start=None,save=True)->pd.DataFrame:
        """Creates a ensemble predictions for a single timestamp and saves it to a zrx files (if save) and returns the dataframe(s)?
        Start is optional and will be taken from the config if not provided.

        Args:
            start (str, optional): The timestamp 1h before the first predicted value. You must have ICON grib files for this timestamp. Defaults to None.
            save (bool, optional): Whether to save the generated prediction. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        cur_start = start or self.config['start']
        df_x, forecast_index = self.data_module.get_forecast_tensor(cur_start,mode='ensemble')
        time_string = pd.to_datetime(cur_start).strftime('%Y%m%d%H')


        df_input_backup = df_x.copy()
        df_agg = pd.DataFrame()
        for i in range(1,21):
            #for column in ens_dict:
            # replace all precipitation columns with the ensemble precipitation
            for column,area in {k:v for k,v in ens_dict.items() if k in df_x.columns}.items():
                faked_prec_index = list(df_x.columns).index(column)

                # overwrite the last 48 values of the current precipitation forecast column with one precipitation forecast from the ensemble
                df_x.iloc[-48:,faked_prec_index] = self._load_prec_zrx(time_string,area,i)

                #we want to backup the ensemble precipitation forecasts
                df_input_backup[f"{column}_{i}" ] = df_x[column]

            x = self.data_module.scaler.transform(df_x)
            x = torch.unsqueeze(torch.tensor(x,dtype=torch.float32), 0)

            y_pred = self.get_pred_manually(x)
            y_pred = pd.DataFrame(y_pred, index=forecast_index,columns=range(1, 49))
            df_zrx = self._to_zrx(y_pred,member=i)

            #aggregate the forecasts in case we want to use it later.
            if i == 1:
                df_agg.index = df_zrx.index
                df_agg['timestamp'] = df_zrx['timestamp']
                df_agg['forecast'] = df_zrx['forecast']
            df_agg[i] = df_zrx['value']

            if save:
                target_folder = self.config['target_file'] if self.config['target_file'].is_dir() else self.config['target_file'].parent
                zrx_name = target_folder / f"{time_string}_{self.config['gauge_name']}_{i}.zrx"

                self._save_zrx(df_zrx,zrx_name)
        if save:
            df_input_backup.to_csv(self.config['input_backup_folder'] / f"{time_string}_{self.config['gauge_name']}_backup.csv")


        return df_agg

    def get_pred_manually(self,x:torch.Tensor) -> torch.Tensor:
        """Badly named. Makes a prediction using forward and inverts scaling.
        Does not use a trainer or dataloader, hence the name.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        # If we use cuda we need to move the tensor to the gpu and the predictions back to the cpu
        if self.cuda:
            x = x.cuda()
        with torch.no_grad():
            y_pred = self.model(x)
            y_pred = y_pred * (self.model.target_max -self.model.target_min) + self.model.target_min
        if self.cuda:
            y_pred = y_pred.cpu()
        return y_pred

    def _to_zrx(self,df:pd.DataFrame,member=1) -> pd.DataFrame:
        """Transforms a dataframe with predictions to a dataframe with the zrx format.
        This means: rounding/flooring; transposing; and having the columns [timestamp, forecast, member, value]

        Args:
            df (pd.DataFrame): prediction dataframe
            member (int, optional): member number in ensembles. Defaults to 1.

        Returns:
            pd.DataFrame: _description_
        """
        #TODO floor or round?
        df_pred_zrx = df.apply(np.floor).astype(int).transpose()
        df_pred_zrx.columns = ['value']
        df_pred_zrx['member'] = member
        df_pred_zrx['forecast'] = pd.period_range(start=df.index[0]+pd.offsets.Hour(1), periods=self.config['out_size'], freq='H')
        df_pred_zrx = df_pred_zrx.astype({'forecast':'datetime64'})
        df_pred_zrx['timestamp'] = df.index[0]
        df_pred_zrx = df_pred_zrx[['timestamp','forecast','member','value']]
        return df_pred_zrx

    def _save_zrx(self,df_zrx:pd.DataFrame,zrx_name:Path):
        """Save the zrx file with predictions to a file.

        Args:
            df_zrx (pd.DataFrame): prediction
            zrx_name (Path): target path
        """
        logging.info("Saving ZRX-prediction to %s",zrx_name.resolve())
        with open(zrx_name , 'w', encoding="utf-8") as file:
            file.write('#REXCHANGEWISKI.' + self.config['gauge_name'] + '.W.KNN|*|\n')
            file.write('#RINVAL-777|*|\n')
            file.write('#LAYOUT(timestamp,forecast, member,value)|*|\n')

        df_zrx.to_csv(path_or_buf = zrx_name,
                    header = False,
                    index=False,
                    mode='a',
                    sep = ' ',
                    date_format = '%Y%m%d%H%M')

    def _load_prec_zrx(self,time_string,area,i) -> pd.DataFrame:
        zrx_file = self.config['ensemble_folder'] / f"{time_string}_{area}_{i}.zrx"
        prec_forecast = pd.read_csv(zrx_file,skiprows=3,sep=' ',usecols=[3],header=None)[3].values
        return prec_forecast


def main(args):
    """Main Function for prediction

    Args:
        args (Namespace): CLI arguments
    """

    config = get_config(args)
    logging.basicConfig(filename=config['log_file'],
        format='%(asctime)s;%(levelname)s;%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    logging.info('Start of this execution======================================================================')
    logging.info("Executing %s with parameters %s ",sys.argv[0],sys.argv[1:])

    predictor = WaVoPredictor(config,cuda=use_cuda)

    if config['mode'] =='single':
        predictor.single()
    elif config['mode'] =='ensemble':
        predictor.ensemble()
    elif config['mode'] =='evaluate':
        predictor.evaluate()
        
    logging.info('End of this execution======================================================================')

if __name__ == "__main__":
    main(parse_args())
