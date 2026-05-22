source .venv/bin/activate
python3 nhits_predict.py --exg_variables wind_speed temperature --model_name temp_wind-speed --particle NO2

python3 nhits_predict.py --exg_variables wind_speed temperature --model_name temp_wind-speed --particle PM2.5

python3 nhits_predict.py --exg_variables wind_speed temperature --model_name temp_wind-speed --particle O3

python3 nhits_predict.py --exg_variables wind_speed temperature --model_name temp_wind-speed --particle PM10

#python3 seasonal_naive.py
