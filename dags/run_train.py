import datetime

from airflow.decorators import task, dag


@task
def read_train_data(n_data=10000):
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
    synthesize_restaurant_df(n_data).to_csv(f'data/train_{timestr}.csv', index=False)
    return timestr


@task
def train(timestr):
    '''
    similar to read_test_data, real use could be specified to write model results to database
    '''
    import pandas as pd
    from restaurant_models.models import save_model, train_dummy_data, get_dummy_models
    from synthesize_restaurant_data.preprocess import prepare_dummy_model_data

    x, y, prep_mean = prepare_dummy_model_data(pd.read_csv(f'data/train_{timestr}.csv'))

    [save_model(v, name=f'models/{timestr}') for v in train_dummy_data(get_dummy_models(), [x, y]).values()]
    prep_mean.to_csv(f'{timestr}_prep_mean.csv')


@dag(dag_id=f'daily_training_run',
     start_date=datetime.datetime(2023, 11, 18), schedule="0 5 * * *",
     default_args={'owner': 'Paul'},
     tags=['daily_training_run'])
def train_dag():
    train(read_train_data())


train_dag()
