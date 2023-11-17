import pandas as pd

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


def get_xy(df, y_col='prep_time_seconds'):
    return df.drop(y_col, axis=1), df[y_col]


if __name__ == '__main__':
    preprocess(synthesize_restaurant_df()).to_csv('test_preprocess_df.csv')
