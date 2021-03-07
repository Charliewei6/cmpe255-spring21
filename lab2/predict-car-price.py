import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        np.random.seed(2)

        n = len(self.df)

        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()

        y_train_orig = df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values

        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)

        del df_train['msrp']
        del df_val['msrp']
        del df_test['msrp']

        return df_train, df_test, df_val, y_val_orig, y_test_orig, y_train, y_test, y_val

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]
    
    def prepare_X(self,df):
        base = ['engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df_num = df[base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

def test() -> None:
    pre_car = CarPrice()
    pre_car.trim()
    df_train, df_test, df_val, y_val_orig,y_test_orig, y_train, y_test, y_val = pre_car.validate()
    
    X_train = pre_car.prepare_X(df_train)
    w_0, w = pre_car.linear_regression(X_train, y_train)
    y_pred = w_0 + X_train.dot(w)
    #print('Train RMSE:', pre_car.rmse(y_train, y_pred))

    X_val = pre_car.prepare_X(df_val)
    y_pred = w_0 + X_val.dot(w)
    #print('Validation RMSE:', pre_car.rmse(y_val, y_pred))

    X_val = pre_car.prepare_X(df_test)
    y_pred = w_0 + X_val.dot(w)
    #print('Test RMSE:', pre_car.rmse(y_test, y_pred))

    df_test['msrp'] = y_test_orig
    df_test['msrp_pred'] = np.expm1(y_pred)
    del df_test['make']
    del df_test['model']
    del df_test['year']
    del df_test['engine_fuel_type']
    del df_test['engine_hp']
    print(df_test[10:15])
if __name__ == "__main__":
    # execute only if run as a script
    test()