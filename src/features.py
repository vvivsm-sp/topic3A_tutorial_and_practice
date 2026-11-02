import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    def parse_storey(s):
        lo, hi = s.split(' TO ')
        return (int(lo) + int(hi)) / 2

    def parse_lease(s):
        parts = s.split()
        years = int(parts[0])
        months = int(parts[2]) if len(parts) > 2 else 0
        return years + months / 12

    df = df.copy()
    df['storey_midpoint'] = df['storey_range'].apply(parse_storey)
    df['remaining_lease_years'] = df['remaining_lease'].apply(parse_lease)
    df = pd.get_dummies(df, columns=['town', 'flat_type', 'flat_model'])
    drop_cols = ['block', 'street_name', 'storey_range', 'remaining_lease', 'month']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def get_train_test_split(df, target='resale_price', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)