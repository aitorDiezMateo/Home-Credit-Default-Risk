import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.impute import KNNImputer


def cargar_datos(ruta):
    return pd.read_csv(ruta)

def forma(df):
    dimensions = df.shape
    return f"{dimensions[0]} filas y {dimensions[1]} columnas"


def format_boolean_columns(df, boolean_columns, true_label='Y', false_label='N'):
    """
    Converts specified boolean columns from custom true/false labels to boolean True/False.

    Parameters:
        df: The DataFrame to be formatted.
        boolean_columns (str or list): A column name or a list of column names that contain custom boolean values.
        true_label: The label representing True (default is 'Y').
        false_label: The label representing False (default is 'N').

    Returns:
        pd.DataFrame: The DataFrame with corrected boolean columns.
    """
    if isinstance(boolean_columns, str):
        boolean_columns = [boolean_columns]

    for col in boolean_columns:
        df[col] = df[col].map({true_label: True, false_label: False})

    return df


def missing_values_percentage(df):
    """
    Calculates the percentage of missing values in each column of a DataFrame.
    Args:
        df: the df frame to analyze

    Returns:
        pd.DataFrame: A DataFrame containing the percentage of missing values in each column.
    """

    missing_percent = df.isnull().mean() * 100
    missing_df = pd.DataFrame(missing_percent, columns=['Missing Percentage'])
    missing_df = missing_df[missing_df['Missing Percentage'] > 0]
    missing_df = missing_df.sort_values(by='Missing Percentage', ascending=False)

    return missing_df

def drop_columns(df, missing_df, default_threshold=50):
    """
    Drops columns with missing values above a certain threshold.

    Args:
        df: DataFrame to drop columns from.
        missing_df: DataFrame with the percentage of missing values in each column
        default_threshold: The threshold above which columns will be dropped (default is 50).

    Returns:
        pd.DataFrame: df DataFrame with columns dropped.
    """
    columns_to_drop = missing_df[missing_df['Missing Percentage'] > default_threshold].index
    df = df.drop(columns=columns_to_drop)

    return df



def impute_with_knn(df, n_neighbors=5):
    """
    Imputes missing values using the K-Nearest Neighbors Imputer algorithm.

    Args:
        df: DataFrame to impute missing values to.
        columns_to_impute: Columns to impute missing values to.
        n_neighbors: Number of neighbors to use for imputation (default is 5).

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    return df_imputed


def splits_creation(data,col):
    train = data[data['is_nan'] == 0]
    test = data[data['is_nan'] == 1]
    X_train = train.drop([col,"is_nan"], axis=1)
    y_train = train[col]
    X_test = test.drop([col,"is_nan"], axis=1)
    return X_train, y_train, X_test

def factorize_categoricals(df):
    for cat_col in df.select_dtypes(include='object'):
        df[cat_col] = pd.factorize(df[cat_col])
    return df

def train_predict(mode, X_train, y_train, X_test):
    if mode == "regression":
        model = LGBMRegressor()
    else:
        model = LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def impute_missing_values(data, cols_list, mode):
    for col in cols_list:
        df = data.copy()
        nan_ixs = np.where(data[col].isna())[0]
        data['is_nan'] = 0
        data.loc[nan_ixs, 'is_nan'] = 1
        X = data.drop([col], axis=1)
        y = data[col]
        X = factorize_categoricals(X)
        X_train, y_train, X_test = splits_creation(data,col)
        y_pred = train_predict(mode, X_train, y_train, X_test)
        df.loc[nan_ixs, col] = y_pred
    return df
