"""
Forecast on hold-out test set using the best NHITS configuration for each particle
"""

import numpy as np 
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, HuberLoss
from neuralforecast.losses.pytorch import MSE
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
from neuralforecast.models import NHITS
import ast
import re

# set CPU fallback for torch when mps - apple silicon GPUs - are not supported
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# define command line argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--exg_variables', nargs='+', help= 'list of exogenous variables to include')
    parser.add_argument('--model_name', type=str, help='name of model combination, must be a row in results df')
    parser.add_argument('--particle', type=str, help='particle to forecast for')
    args = vars(parser.parse_args())
    
    return args

def refit_and_predict(best_config, df, model_name, exg_variables, particle):

    """
    Refit NHITS model on the full dataset based on the best model configurations and predicts on the hold-out test set.
    Saves forecast to folder.

    """

    horizon = 96  # forecast 96 hours ahead
    
    # rename + add cols to fit with neuralforecast expectations
    df = df.rename(columns={'Recorded': 'ds', f'{particle}': 'y'})
    df['unique_id'] = particle

    particle_df = df[['ds', 'unique_id', 'y'] + exg_variables]

    # divide in train + test dfs
    train_df = particle_df.iloc[:-horizon]
    test_df  = particle_df.iloc[-horizon:]
    test_df = test_df[['ds', 'unique_id'] + exg_variables]

    # hacky way of removing MSE() so it's not parsed as a string literal when loading the best model configs..
    best_config = ast.literal_eval(re.sub(r"MSE\(\)", "None", best_config))
    best_config = {k: v for k, v in best_config.items() if k not in {'loss', 'valid_loss', 'h'}}
    
    # specify model with best configs
    model = NHITS(
        h = horizon,
        **best_config
    )

    # fit and predict
    nf = NeuralForecast(models=[model], freq="h")
    
    nf.fit(train_df)
    forecast = nf.predict(futr_df=test_df)

    # save to folder
    forecast_out_path = os.path.join('out', 'forecasts', 'NHITS')
    os.makedirs(forecast_out_path, exist_ok=True)
    forecast.to_csv(os.path.join(forecast_out_path, f"{particle}_{model_name}_forecast.csv"))


def main():

    # load command line args
    args = argument_parser()

    # read data and convert to datetime
    df = pd.read_csv(os.path.join('data', 'aarhus_air_quality.csv'))
    df['Recorded'] = pd.to_datetime(df['Recorded'])
        
    exg_variables = args['exg_variables']

    # read results df:
    results_path = os.path.join('out', 'cv_results', 'NHITS', f"{args['particle']}_cv_results.csv")
    results_df = pd.read_csv(results_path)

    best_config = results_df.query(f"model_type == '{args['model_name']}'")['best_config'].iloc[0]

    # refit model and predict on hold out test set
    refit_and_predict(best_config, df, args['model_name'], exg_variables, args['particle'])

if __name__ == '__main__':
    main()


