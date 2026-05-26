# air-quality-forecasting
Exam project for the course 'Data Science, Prediction and Forecasting' @ Aarhus University, 2026


## Project Structure

```
air-quality-forecasting/
│
├── data/
│   └── aarhus_air_quality.csv           # 
│
├── nbs/                                 # Jupyter Notebooks
│   ├── ARIMA_tune_fit_CV.ipynb          # Grid search hyperparameters and find best ARIMA models 
│   ├── CCF_plots.ipynb                  # Plot CCF for pollutants
│   ├── exogenous_variables.ipynb        # 
│   ├── explore_data_and_ACF.ipynb       # Explores raw pollutant levels and seasonality
│   └── forecast_plots.ipynb             #
│
├── src/                                 # Source code
│   ├── ARIMA_CV.py                      # 
│   ├── ARIMA_tuning.py                  # 
│   ├── nhits_predict.py                 # 
│   ├── nhits.py                         # 
│   └── seasonal_naive.py                # 
│
├── out/                                 # plots and results
│   ├── cv_plots/                        # cv plots for the NHITS
│   ├── cv_results/                      # cv results for ARIMA and NHITS
│   ├── forecasts.png                    # hold-out forecasts for all models
│   └── plots                            # 
│
├── .env                                 # 
├── README.md                            #   
├── requirements.txt                     # Python dependencies
├── run_forecasts.sh                     # Run NHITS modelling
├── env_to_jupyter.sh                    # creates kernel from .venv to be used for jupyter 
└── setup.sh                             # Set up virtual environment and install required packages
```