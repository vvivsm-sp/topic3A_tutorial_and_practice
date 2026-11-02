# Vivian 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from src.features import load_data, engineer_features, get_train_test_split


def train_and_evaluate(data_path: str):
    df = load_data(data_path)
    df = df[df['month'] < '2022-01']
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: -x[1]
    )

    print(
        f'Random Forest — '
        f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f} '
        f'MAE: {mean_absolute_error(y_test, y_pred):,.0f} '
        f'R²: {r2_score(y_test, y_pred):.3f}'
    )

    print(f'Top 5 features: {importances[:5]}')


if __name__ == '__main__':
    train_and_evaluate('data/raw/hdb_resale_prices.csv')