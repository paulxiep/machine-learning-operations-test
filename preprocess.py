from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split

from synthesize_data import synthesize_restaurant_df


def preprocess(df):
    return pd.concat([
        pd.concat([
            df['order_acknowledged_at'].dt.hour,
            df['order_acknowledged_at'].dt.weekday,
            df['order_acknowledged_at'].dt.day,
            df['restaurant_id'].map(df.restaurant_id.value_counts())
        ], axis=1, keys=['hour', 'weekday', 'monthday', 'r_counts']),
        df], axis=1
    )[['order_value_gbp',
       'number_of_items',
       'r_counts',
       'monthday',
       'hour',
       'weekday',
       'city',
       'country',
       'type_of_food',
       'restaurant_id',
       'prep_time_seconds']]


def post_split_process(df_train, df_test, prep_mean=None):
    @lru_cache
    def get_prep_mean():
        if prep_mean is None:
            return df_train.groupby('restaurant_id').prep_time_seconds.mean()
        else:
            return prep_mean

    return tuple([*get_xy(df_train.merge(get_prep_mean(),
                                         how='left', on='restaurant_id', suffixes=('', '1'))),
                  *get_xy(df_test.merge(get_prep_mean(),
                                        how='left', on='restaurant_id', suffixes=('', '1'))), get_prep_mean()])


def get_xy(df, y_col='prep_time_seconds'):
    return df.drop(y_col, axis=1), df[y_col]


def prepare_data(prep_mean=None):
    return post_split_process(*train_test_split(preprocess(synthesize_restaurant_df()), random_state=0),
                              prep_mean=prep_mean)


if __name__ == '__main__':
    preprocess(synthesize_restaurant_df()).to_csv('test_preprocess_df.csv')
