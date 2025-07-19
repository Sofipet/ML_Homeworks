import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Optional, Tuple


def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder]:
    """
    Попередня обробка даних: видалення зайвих колонок, кодування, опціональне масштабування.

    Args:
        raw_df (pd.DataFrame): Сирий датафрейм.
        scale_numeric (bool): Чи масштабувати числові ознаки.

    Returns:
        X_train, y_train, X_val, y_val, input_cols, scaler, encoder
    """
    df = raw_df.copy()
    df = df.drop(columns=["id", "CustomerId", "Surname"], errors="ignore")
    df = df.dropna(subset=["Exited"])

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Категоріальні та числові ознаки
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    numeric_cols = [col for col in X_train.columns if X_train[col].dtype != "object"]

    # Кодування
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical_cols])

    def encode(df_: pd.DataFrame) -> pd.DataFrame:
        encoded = encoder.transform(df_[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df_.index)
        return pd.concat([df_.drop(columns=categorical_cols), encoded_df], axis=1)

    X_train = encode(X_train)
    X_val = encode(X_val)

    # Масштабування
    scaler = None
    if scale_numeric:
        scaler = StandardScaler()
        scaler.fit(X_train[numeric_cols])
        X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    input_cols = X_train.columns.tolist()

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[StandardScaler] = None
) -> pd.DataFrame:
    """
    Обробка нових (тестових) даних для передбачення.

    Args:
        new_df (pd.DataFrame): Нові дані (наприклад, test.csv).
        input_cols (List[str]): Список колонок, які були у train.
        encoder (OneHotEncoder): Навчений енкодер.
        scaler (Optional[StandardScaler]): Навчений скейлер або None.

    Returns:
        pd.DataFrame: Оброблений тестовий датафрейм.
    """
    df = new_df.copy()
    df = df.drop(columns=["id", "CustomerId", "Surname"], errors="ignore")

    categorical_cols = encoder.feature_names_in_.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    if scaler is not None:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Вирівнювання колонок
    for col in input_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[input_cols]

    return df
