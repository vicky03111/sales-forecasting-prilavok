import pandas as pd
import numpy as np


def create_temporal_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    return df


def create_avg_sales_feature(df):
    df = df.copy()
    df = df.sort_values(by=['store', 'dept', 'date']).reset_index(drop=True)
    df['avg_sales_before'] = (
        df.groupby(['store', 'dept'])['weekly_sales']
        .apply(lambda x: x.expanding().mean().shift(1))
        .reset_index(level=[0, 1], drop=True)
    )
    return df


def create_lag_features(df):
    df = df.copy()
    df = df.sort_values(by=['store', 'dept', 'date']).reset_index(drop=True)
    df['sales_1week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(1)
    df['sales_2week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(2)
    df['sales_4week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(4)
    return df


def create_rolling_features(df):
    df = df.copy()
    df = df.sort_values(by=['store', 'dept', 'date']).reset_index(drop=True)

    df['sales_shifted_for_rolling'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(1)

    df['mean_sales_2week'] = (
        df.groupby(['store', 'dept'])['sales_shifted_for_rolling']
        .transform(lambda x: x.rolling(window=2, min_periods=2).mean())
    )

    df['mean_sales_4week'] = (
        df.groupby(['store', 'dept'])['sales_shifted_for_rolling']
        .transform(lambda x: x.rolling(window=4, min_periods=4).mean())
    )

    df.drop(columns=['sales_shifted_for_rolling'], inplace=True)
    return df


def calculate_psi(actual, expected, num_buckets=10):
    actual = actual[~np.isnan(actual)]
    expected = expected[~np.isnan(expected)]

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(actual, percentiles)

    buckets[0] -= 1e-5
    buckets[-1] += 1e-5

    actual_counts, _ = np.histogram(actual, bins=buckets)
    expected_counts, _ = np.histogram(expected, bins=buckets)

    actual_pcts = actual_counts / len(actual)
    expected_pcts = expected_counts / len(expected)

    actual_pcts = np.where(actual_pcts == 0, 1e-4, actual_pcts)
    expected_pcts = np.where(expected_pcts == 0, 1e-4, expected_pcts)

    psi_value = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return psi_value


def interpret_psi(psi):
    if psi < 0.1:
        return 'PSI < 0.1 — стабилен, можно использовать'
    elif psi < 0.2:
        return '0.1 <= PSI < 0.2 — умеренный дрейф, требуется внимание'
    else:
        return 'PSI >= 0.2 — значительный дрейф, признак нестабилен'