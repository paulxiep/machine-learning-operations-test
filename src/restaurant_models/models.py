import pickle
from functools import lru_cache

from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from synthesize_restaurant_data.preprocess import prepare_dummy_model_data
from synthesize_restaurant_data.generate_synthetic_data import synthesize_restaurant_df


def functional_model(model):
    '''
    wrap any machine learning model and override fit method to conform to functional programming style,
    due to compatibility restriction with sklearn, positional argument is not allowed in __init__
    '''

    class FunctionalModel(model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, *args, **kwargs):
            super().fit(*args, **kwargs)
            return self

        def load_model(self, file_name):
            super().load_model(file_name)
            return self

        def parent(self):
            return self.__class__.__bases__[0].__name__

        def pickle(self, path):
            self.__class__ = self.__class__.__bases__[0]
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    return FunctionalModel


# get a stacking ensemble of models
def get_stacking_models(gbr, rf, cat, xgb, final):
    '''
    copied from provided jupyter, but wrapped functional_model over result
    '''
    # define the base models
    level0 = list()
    level0.append(('gbr', gbr))
    level0.append(('rf', rf))
    level0.append(('cat', cat))
    level0.append(('xgb', xgb))
    # define meta learner model
    level1 = final
    # define the stacking ensemble
    model = functional_model(StackingRegressor)(estimators=level0, final_estimator=level1, cv=3)
    return model


def get_dummy_models():
    @lru_cache
    def get_base_models():
        return {
            'rf': functional_model(RandomForestRegressor) \
                (n_estimators=100, min_samples_split=6, max_features='log2'),
            'gbr': functional_model(GradientBoostingRegressor) \
                (n_estimators=100, learning_rate=0.1, min_samples_split=2),
            'cat': functional_model(CatBoostRegressor) \
                (depth=6, iterations=100, learning_rate=0.1, silent=True),
            'xgb': functional_model(XGBRegressor) \
                (max_depth=4, n_estimators=100, learning_rate=0.1)
        }

    return {
        **get_base_models()  # ,
        # 'stacking': get_stacking_models(**get_base_models(), final=Ridge(alpha=0.3))
    }


def train_dummy_data(dummy_models, dummy_data=prepare_dummy_model_data(synthesize_restaurant_df())[:2]):
    return {k: v.fit(*dummy_data) for k, v in dummy_models.items()}


def save_model(model, name='dummy'):
    if isinstance(model, CatBoostRegressor):
        model.save_model(f'{name}_cat.cbm')
    elif isinstance(model, XGBRegressor):
        model.save_model(f'{name}_xg.json')
    else:
        model.pickle(f'{name}_{model.parent()}.pkl')


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_models(name='dummy'):
    return {
        'rf': load_pickle(f'{name}_RandomForestRegressor.pkl'),
        'gbr': load_pickle(f'{name}_GradientBoostingRegressor.pkl'),
        'cat': functional_model(CatBoostRegressor)().load_model(f'{name}_cat.cbm'),
        'xgb': functional_model(XGBRegressor)().load_model(f'{name}_xg.json')
    }


if __name__ == '__main__':
    dummy_data = prepare_dummy_model_data()
    [save_model(v) for v in train_dummy_data(get_dummy_models(), dummy_data[:2]).values()]
    dummy_data[2].to_csv('prep_mean.csv')
