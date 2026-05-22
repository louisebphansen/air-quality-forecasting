from neuralforecast.losses.numpy import smape
import numpy as np
import os
import pandas as pd

def lag24_predict(y_before_test, horizon=96, lag=24, repeats=4):
    last_cycle = np.array(y_before_test[-lag:])
    return np.tile(last_cycle, repeats)[:horizon]

def main():
    df = pd.read_csv(os.path.join('aarhus_air_quality.csv'))
    df['Recorded'] = pd.to_datetime(df['Recorded'])

    test_size = 96

    train_df = df.iloc[:-test_size]
    test_df  = df.iloc[-test_size:]

    for particle in ['NO2', 'O3', 'PM2.5', 'PM10']:

        y_before_test = train_df[particle][-24:]

        y_pred = lag24_predict(y_before_test)

        eval_df = pd.DataFrame({
            'unique_id': particle,
            'y': test_df[particle].values,
            'lag24': y_pred
        })

        #result = smape(eval_df, models=['lag24'], id_col='unique_id', target_col='y')
        #print(f"{particle}: {result}")

        results_outpath = os.path.join('out', 'forecasts', 'naive')
        os.makedirs(results_outpath, exist_ok=True)

        eval_df.to_csv(os.path.join(results_outpath, f'{particle}_naive.csv'))

if __name__ == '__main__':
    main()




