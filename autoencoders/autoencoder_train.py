from autoencoders import Conv1dAutoEncoder, LSTMAutoEncoder, TickerDataModule, MLPAutoEncoder

from pytorch_lightning import Trainer, loggers
import pandas as pd


def test_conv_encoder(data_path, log_path):
    model = Conv1dAutoEncoder(1, 100)
    tb_logger = loggers.TensorBoardLogger(log_path, name='')
    trainer = Trainer(gpus=1, max_epochs=100, logger=tb_logger)

    dm = TickerDataModule(data_path, preprocessing=False, time_period=100)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_lstm_encoder(data_path, log_path):
    model = LSTMAutoEncoder(100, 1, embedding_dim=100)
    tb_logger = loggers.TensorBoardLogger(log_path, name='lstm')
    trainer = Trainer(gpus=1, max_epochs=100, logger=tb_logger)

    dm = TickerDataModule(data_path, batch_size=18, time_period=100, preprocessing=False)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_mlp_encoder(data_path,log_path):
    model = MLPAutoEncoder(100, 100)
    tb_logger = loggers.TensorBoardLogger(log_path, name='mlp')
    trainer = Trainer(gpus=1, max_epochs=150, logger=tb_logger)

    dm = TickerDataModule(data_path,
                          preprocessing=False,
                          time_period=100)

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':
    experiment_name = 'usa_2012'
    path = f'../data/data/usa_2012/ticker_data_preprocessed.csv'
    
    import pandas as pd
    pd.read_csv(path)
    log_path = f'lightning_logs_{experiment_name}'
    test_lstm_encoder(path, log_path)
    test_conv_encoder(path, log_path)
    test_mlp_encoder(path, log_path)
