# Forecasting Air Quality in Aarhus: Comparing Time-Series Models

This repository contains the code for our exam project for the course 'Data Science, Prediction and Forecasting' @ Aarhus University, 2026

Code for the project was equally contributed to by Ida Munch Andresen (IMA) and Louise Brix Pilegaard Hansen (LBPH).

## Project Structure

```
air-quality-forecasting/
│
├── data/
│   └── aarhus_air_quality.csv           # dataset
│
├── nbs/                                 # Jupyter Notebooks
│   ├── ARIMA_tune_fit_CV.ipynb          # Grid search hyperparameters and find best ARIMA models 
│   ├── CCF_plots.ipynb                  # Plot CCF for pollutants
│   ├── exogenous_variables.ipynb        # Explores correlations between exogenous variables
│   ├── explore_data_and_ACF.ipynb       # Explores raw pollutant levels and seasonality
│   └── forecast_plots.ipynb             # Code for plotting and saving forecasting results
│
├── src/                                 # Source code
│   ├── ARIMA_CV.py                      # 
│   ├── ARIMA_tuning.py                  # 
│   ├── nhits_predict.py                 # predict on hold-out test set using NHITS
│   ├── nhits.py                         # fit NHITS models with exogenous variables on train data
│   └── seasonal_naive.py                # predict using seasonal naive model
│
├── out/                                 # plots and results
│   ├── cv_plots/                        # cv plots for the NHITS
│   ├── cv_results/                      # cv results for ARIMA and NHITS
│   ├── forecasts.png                    # hold-out forecasts for all models
│   └── plots                            # result plots
│
├── README.md                            # 
├── requirements.txt                     # Python dependencies
├── run_forecasts.sh                     # Run NHITS modelling
├── env_to_jupyter.sh                    # creates kernel from .venv to be used for jupyter 
└── setup.sh                             # Set up virtual environment and install required packages
```

## Prerequisites

Clone the project's repository with:

```
git clone https://github.com/louisebphansen/air-quality-forecasting.git
```

### Data and usage

#### Data

Data from OpenMeteos public [API] (https://open-meteo.com/en/docs/air-quality-api) for the specified period (18th of April 2025 - 18th of April 2026) are gathered and saved to a CSV file under ```data```. 

#### Usage

To create the virtual environment, make sure to install the *venv* package.

Next, set up the environment and install the required packages with:

```
bash setup.sh
```

Optionally, run 


```
env_to_jupyter.sh
```

to install .venv to kernel for Jupyter Notebooks. 

All source code scripts (in ```src```) should be run from the terminal from the main folder.

Specifically, to run predictions for NHITS on the hold-out test set, run this bash script:

```
bash run_forecasts.sh 
```