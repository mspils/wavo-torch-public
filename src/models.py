"""
Neural Networks and LightningModules
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    """LSTM Network

    Args:
        feature_count (int): how many input columns
        in_size (int, optional): input length. Defaults to 144.
        out_size (int, optional): output length. Defaults to 48.
        hidden_size_lstm (int, optional): _description_. Defaults to 128.
        hidden_size (int, optional): _description_. Defaults to 64.
        num_layers (int, optional): _description_. Defaults to 2.
        dropout (float, optional): _description_. Defaults to 0.2.
    """

    def __init__(self, feature_count, in_size=144, out_size=48, hidden_size_lstm=128, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        # note: dropout is not used in the last layer, so only works in num layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0

        self.lstm = nn.LSTM(input_size=feature_count, hidden_size=hidden_size_lstm,
                            num_layers=num_layers, batch_first=True, dropout=lstm_dropout)  # lstm
        self.fc_1 = nn.Linear(hidden_size_lstm, hidden_size)  # fully connected
        # fully connected last layer
        self.dropout = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(hidden_size*in_size, out_size)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        """forward pass

        Args:
            x (Tensor|Array): input data

        Returns:
            _type_: _description_
        """
        output, _ = self.lstm(x)  # (input, hidden, and internal state)
        out = self.relu(output)
        out = self.fc_1(out)  # first dense
        out = self.dropout(out)  # dropout
        out = self.relu(out)  # relu
        out = self.flatten(out)  # flatten
        out = self.fc_2(out)  # final output
        return out


class WaVoLightningModule(pl.LightningModule):
    """LightningModule that defines the training and other steps, logging and the neural network.
    Might need some tweaking for hyperparameter optimization

    Args:
        mean (float): mean of target values
        scale (float): std of target values
        threshold (float): flood threshold
        feature_count (int): how many input columns
        in_size (int, optional): input length. Defaults to 144.
        out_size (int, optional): output length. Defaults to 48.
        hidden_size_lstm (int, optional): _description_. Defaults to 128.
        hidden_size (int, optional): _description_. Defaults to 64.
        num_layers (int, optional): _description_. Defaults to 2.
        dropout (float, optional): _description_. Defaults to 0.2.
        learning_rate (float, optional): _description_. Defaults to 0.001.
    """

    # pylint: disable-next=unused-argument
    def __init__(self, target_min,target_max, feature_count, in_size=144, out_size=48, hidden_size_lstm=128, hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001):
        #def __init__(self, target_scaler, feature_count, in_size=144, out_size=48, hidden_size_lstm=128, hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001):
        #def __init__(self, mean, scale, feature_count, in_size=144, out_size=48, hidden_size_lstm=128, hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001):


        super().__init__()
        target_min = torch.tensor(target_min, dtype=torch.float32) if isinstance(target_min,float) else target_min
        target_max = torch.tensor(target_max, dtype=torch.float32) if isinstance(target_max,float) else target_max

        self.save_hyperparameters(ignore=['target_min','target_max'])
        self.save_hyperparameters({'target_min':target_min.item(),'target_max':target_max.item()})

        self.target_min = target_min
        self.target_max = target_max
        self.in_size = in_size
        self.out_size = out_size
        self.lstm = LSTM(
            feature_count=feature_count,
            in_size=in_size,
            out_size=out_size,
            hidden_size_lstm=hidden_size_lstm,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    # pylint: disable-next=arguments-differ
    def forward(self, x):
        return self.get_network()(x)

    # pylint: disable-next=unused-argument, arguments-differ
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.lstm(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("hp/train_loss", loss)
        return loss

    # pylint: disable-next=unused-argument, arguments-differ
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.lstm(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        # syncdist needed for optuna hyperparameter optimization to work
        self.log("hp/val_loss", val_loss, sync_dist=True)

    # pylint: disable-next=unused-argument, arguments-differ
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.lstm(x)
        test_loss = nn.functional.mse_loss(y_hat, y)
        self.log("hp/test_loss", test_loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch

        #This is not great, but necessary if i want to used a scaler.
        pred = self.lstm(x)#.detach().cpu()

        pred = pred * (self.target_max-self.target_min) + self.target_min


        # TODO check if i can just convert this to a Tensor earlier. Lightning should move it?
        #if batch_idx == 0 and isinstance(self.mean, float):
        #    device = torch.device(pred.get_device())
        #    self.mean = torch.tensor(self.mean, device=device)
        #    self.scale = torch.tensor(self.scale, device=device)
        #pred = pred*self.scale + self.mean


        return pred

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def get_network(self) -> nn.Module:
        """Just returns the actual neural network.
        Kinda unnecessary right now, but might be useful later if i want to use different networks
        Returns:
            nn.Module: _description_
        """
        return self.lstm
