import numpy as np 
from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, HuberLoss
from ray import tune # hyperparameter tuning package
from neuralforecast.losses.pytorch import MSE
import matplotlib.pyplot as plt
import os
import pandas as pd

# set CPU fallback for torch when mps - apple silicon GPUs - are not supported
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def run_model(particle, exog_variables, nhits_config, df_train, step_size, w_size, model_save_name):

    horizon = 96          # forecast 96 hours ahead
    num_samples = 5       # number of Ray Tune trials (5 times, it randomly selects a combination of these values)

    nhits_config = nhits_config.copy()

    # if there's any exogenous variables, add to nhits model config
    if exog_variables:
        nhits_config["futr_exog_list"] = tune.choice([exog_variables])

    # rename + add cols to fit with neuralforecast expectations
    df = df_train.rename(columns={'Recorded': 'ds', f'{particle}': 'y'})
    df['unique_id'] = particle

    particle_df = df[['ds', 'unique_id', 'y', 'wind_speed', 'temperature', 'humidity', 'solar_radiation']]

    # specify model
    model = AutoNHITS(
        h=horizon,
        loss=MSE(),
        valid_loss=MSE(),
        config=nhits_config,
        num_samples=num_samples,
        refit_with_val=True
    )

    nf = NeuralForecast(models=[model], freq="h")

    # set up cross validation; we're doing hyperparameter tuning + cross validation in one
    cv_df = nf.cross_validation(df=particle_df, n_windows=w_size,
                            step_size=step_size, refit=False, verbose=False) # refit = False for computational purposes
    
    # get measurements
    mae = np.mean(np.abs(cv_df["AutoNHITS"] - cv_df["y"]))
    mse = np.mean((cv_df["AutoNHITS"] - cv_df["y"]) ** 2)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.4f}")

    # get cutoffs for expanding windows (saved by the cross validation function)
    cutoffs = cv_df["cutoff"].unique()

    # create + save plot

    fig, axes = plt.subplots(len(cutoffs), 1, figsize=(14, 4 * len(cutoffs)))

    for ax, cutoff in zip(axes, cutoffs):
        window_df = cv_df[cv_df["cutoff"] == cutoff].copy()
        
        # Plot a bit of training context before the cutoff
        context_df = particle_df[particle_df["ds"] > cutoff - pd.Timedelta(hours=horizon*2)]
        context_df = context_df[context_df["ds"] <= cutoff]
        
        ax.plot(context_df["ds"], context_df["y"], 
                color="steelblue", label="Historical")
        ax.plot(window_df["ds"], window_df["y"], 
                color="steelblue", linestyle="--", alpha=0.5, label="Ground truth")
        ax.plot(window_df["ds"], window_df["AutoNHITS"], 
                color="tomato", linestyle="--", label="AutoNHITS forecast")
        ax.axvline(x=cutoff, color="black", linestyle=":", label="Cutoff")
        
        window_mae = np.mean(np.abs(window_df["AutoNHITS"] - window_df["y"]))
        ax.set_title(f"Cutoff: {cutoff}  |  MAE: {window_mae:.4f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("NO2")
        ax.legend()

    plt.suptitle(f"AutoNHITS CV Results  |  Overall MAE: {mae:.4f}  |  MSE: {mse:.4f}", 
                fontsize=13, y=1.01)
    plt.tight_layout()

    outpath_plots = os.path.join('out', 'cv_plots', 'NHITS_CONFIGS_TESTER', particle)
    os.makedirs(outpath_plots, exist_ok=True)
    plt.savefig(os.path.join(outpath_plots, f"{model_save_name}.png"))

    window_rmses = []
    for cutoff in cutoffs:
        window_df = cv_df[cv_df["cutoff"] == cutoff]
        window_mse = np.mean((window_df["AutoNHITS"] - window_df["y"]) ** 2)
        window_rmses.append(np.sqrt(window_mse))

    rmse = np.mean(window_rmses)
    rmse_sd = np.std(window_rmses)

    best_config = nf.models[0].results.get_best_result().config

    return best_config, rmse, rmse_sd

def main():

    df = pd.read_csv(os.path.join('aarhus_air_quality.csv'))

    # convert back to datetime
    df['Recorded'] = pd.to_datetime(df['Recorded'])
    test_size = 96
    train_df = df.iloc[:-test_size]
    test_df  = df.iloc[-test_size:]
    
   # config = {
    #    "learning_rate": tune.loguniform(1e-4, 1e-2), # step size for learning rate optimizer
     #   "max_steps": tune.choice([2000]), # number of gradient steps (??)
      #  "input_size": tune.choice([7*24]),  # 8 days vs 28 days lookback period
       # "batch_size": tune.choice([1]), # just 1 since i only have one time series
        #"windows_batch_size": tune.choice([256]), # number of windows sampled from the series per training step
     #   "n_pool_kernel_size": tune.choice([[4, 2, 1]]), # maxpool kernel size
    #    "n_freq_downsample": tune.choice([
    #[4, 2, 1],   # coarse/medium/fine decomposition aligned to 96h horizon
        #[24, 4, 1],    # daily cycle emphasis
    #]), # controls interpolation ratio when projecting each stack's output back up to the horizon
   #     "dropout_prob_theta": tune.choice([0.2]), # dropout, regularization to the MLP output
     #   "activation": tune.choice(["ReLU"]), # activation function used in the mlp units
      #  "n_blocks": tune.choice([[3, 3, 3]]), # number of blocks per stack
       # "mlp_units": tune.choice([[[512, 512], [512, 512], [512, 512]]]), # size of MLP units (one pair per stack)
        #"interpolation_mode": tune.choice(["linear"]), # how the coefficents from MLP are interpolated to the full horizon length again
        #"val_check_steps": tune.choice([100]), # how often to evaluate on the validation set during training (??)
        #"random_seed": tune.randint(1, 10) # vary random seet to reduce risk of lucky/unlucky init
    #}

    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "max_steps": tune.choice([2000]),
        "input_size": tune.choice([7*24]),
        "batch_size": tune.choice([1]),
        "windows_batch_size": tune.choice([256]),
        "n_pool_kernel_size": tune.choice([
            [4, 2, 1],
            [2, 1, 1],
            [1, 1, 1],
        ]),
        "n_freq_downsample": tune.choice([
            [4, 2, 1],
            [24, 4, 1],
            [168, 24, 1],
        ]),
        "dropout_prob_theta": tune.choice([0.0, 0.2]),
        "activation": tune.choice(["ReLU"]),
        "n_blocks": tune.choice([[3, 3, 3]]),
        "mlp_units": tune.choice([[[512, 512], [512, 512], [512, 512]]]),
        "interpolation_mode": tune.choice(["linear"]),
        "val_check_steps": tune.choice([100]),
        "random_seed": tune.randint(1, 10)
    }

    model_setup = { 
        'NO2': {
            #'no_exg': None, 
            #'temp': ['temperature'],
            #'wind_speed': ['wind_speed'],
            'temp_wind-speed': ['wind_speed', 'temperature'],
            #'everything': ['wind_speed', 'temperature', 'humidity', 'solar_radiation'] 
    },
        'O3': {
            'no_exg': None, 
            'temp': ['temperature'],
            'wind_speed': ['wind_speed'],
            'temp_wind-speed': ['wind_speed', 'temperature'],
            'everything': ['wind_speed', 'temperature', 'humidity', 'solar_radiation'] 
        },
        'PM2.5':{
            'no_exg': None, 
            'temp': ['temperature'],
            'temp_humidity': ['temperature', 'humidity'],
            'everything': ['wind_speed', 'temperature', 'humidity', 'solar_radiation'] 
        },
        'PM10':{
            'no_exg': None, 
            'temp': ['temperature'],
            'temp_humidity': ['temperature', 'humidity'],
            'everything': ['wind_speed', 'temperature', 'humidity', 'solar_radiation'] 
        },
    }


    #for particle in ['NO2', 'O3', 'PM2.5', 'PM10']:
    #for particle in ['PM2.5', 'PM10']:
    for particle in ['NO2']:
        rows = []
        for name, setup in model_setup[particle].items():
            best_config, rmse, rmse_sd = run_model(particle, setup, config, train_df, 96, 5, name)

            rows.append({
                'particle': particle,
                'model_type': name,
                'rmse': rmse,
                'rmse_sd': rmse_sd,
                'best_config': best_config
            })

        results_df = pd.DataFrame(rows).sort_values('rmse')

        # save to file 
        results_out_path = os.path.join('out', 'cv_results', 'NHITS')
        os.makedirs(results_out_path, exist_ok=True)
        results_df.to_csv(os.path.join(results_out_path, f"{particle}_cv_results.csv"))

        print(results_df.head())

        # grab best model config + combo
        best_result = results_df.iloc[0]

        print(f"Best model is: {best_result['model_type']}")

if __name__ == '__main__':
   main()




            


    


