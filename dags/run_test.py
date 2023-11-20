import datetime

from airflow.decorators import task, dag


@task
def read_test_data():
    '''
    real use should be something like querying daily data table
    but for the purpose of this mock test, this function will instead call synthetic data generation

    Instead of generating time string at runtime, it can be received as argument from Airflow's scheduler

    NOTE: with airflow module imports inside function is common practice,
    as airflow constantly reloads DAG code, putting import inside functions restricts
    the module import until DAG is actually run
    '''
    from synthesize_restaurant_data.generate_synthetic_data import synthesize_restaurant_df
    timestr = datetime.date.today().strftime("%Y%m%d")
    synthesize_restaurant_df(100).to_csv(f'results/test_{timestr}.csv', index=False)
    return timestr


@task
def predict(timestr):
    '''
    similar to read_test_data, real use could be specified to write model results to database
    '''
    import os
    import pandas as pd
    from restaurant_models.models import load_models
    from synthesize_restaurant_data.preprocess import prepare_test_data
    data = pd.read_csv(f'results/test_{timestr}.csv')
    if os.path.exists(f'models/{timestr}_cat.cbm'):
        models = load_models(f'models/{timestr}')
    else:
        models = load_models('models/dummy')
    for k, v in models.items():
        pd.DataFrame(v.predict(prepare_test_data(data, pd.read_csv('models/prep_mean.csv')) \
                               .drop(['prep_time_seconds'], axis=1, errors='ignore')), columns=['predicted_prep_time']) \
            .to_csv(f'results/{k}_pred_result_{datetime.date.today().strftime("%Y%m%d")}.csv', index=False)


@dag(dag_id=f'daily_prediction_run',
     start_date=datetime.datetime(2023, 11, 18), schedule="0 7 * * *",
     default_args={'owner': 'Paul'},
     tags=['daily_prediction_run'])
def prediction_dag():
    predict(read_test_data())


prediction_dag()
