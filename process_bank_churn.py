import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List, Optional


def split_data(df: pd.DataFrame, target_column: str = "Exited") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Розбиває дані на тренувальні та валідаційні множини.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def encode_categorical(
    X_train: pd.DataFrame, X_val: pd.DataFrame, categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    One-Hot Encoding для категоріальних ознак.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical_cols])
    train_encoded = encoder.transform(X_train[categorical_cols])
    val_encoded = encoder.transform(X_val[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    X_train_encoded = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
    X_val_encoded = pd.DataFrame(val_encoded, columns=encoded_cols, index=X_val.index)

    X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
    X_val = pd.concat([X_val.drop(columns=categorical_cols), X_val_encoded], axis=1)

    return X_train, X_val, encoder, encoded_cols.tolist()


def scale_numerical(
    X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Масштабування числових ознак за допомогою StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    X_train_scaled = scaler.transform(X_train[numeric_cols])
    X_val_scaled = scaler.transform(X_val[numeric_cols])

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=numeric_cols, index=X_val.index)

    X_train = pd.concat([X_train_scaled_df, X_train.drop(columns=numeric_cols)], axis=1)
    X_val = pd.concat([X_val_scaled_df, X_val.drop(columns=numeric_cols)], axis=1)

    return X_train, X_val, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder]:
    """
    Головна функція для попередньої обробки даних.
    """
    df = raw_df.copy()

    # Видаляємо колонку Surname
    if "Surname" in df.columns:
        df.drop(columns=["Surname"], inplace=True)

    # Train-val split
    X_train, y_train, X_val, y_val = *split_data(df),  # розпаковка двійок

    # Визначаємо типи ознак
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Кодування категоріальних
    X_train, X_val, encoder, encoded_cols = encode_categorical(X_train, X_val, categorical_cols)

    # Масштабування числових (опціонально)
    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numerical(X_train, X_val, numeric_cols)

    input_cols = X_train.columns.tolist()

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Обробка нових (тестових) даних перед передбаченням.
    """
    df = new_df.copy()
    if "Surname" in df.columns:
        df.drop(columns=["Surname"], inplace=True)

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Кодування
    encoded = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

    # Масштабування
    if scaler:
        scaled = scaler.transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)
        df.update(scaled_df)

    # Вирівнюємо колонки з тренувальними
    df = df.reindex(columns=input_cols, fill_value=0)

    return df
