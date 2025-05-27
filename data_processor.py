import pandas as pd
import numpy as np
import scipy
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections.abc import Iterable
from typing import Union


def apply_winsorization(data: np.ndarray|pd.Series) -> np.ndarray|pd.Series:
    z_score = zscore(data)
    mean = data.mean()
    std = data.std()
    upper_boundary = mean + std * 3
    lower_boundary = -(mean + std * 3)
    data[z_score > 3] = upper_boundary
    data[z_score < -3] = lower_boundary
    return data


def apply_log(data: np.ndarray|pd.Series) -> np.ndarray|pd.Series:
    log_data = np.log1p(data)
    fig, axes = plt.subplots(1, 2)
    sm.ProbPlot(data).qqplot(ax=axes[0])
    sm.ProbPlot(log_data).qqplot(ax=axes[1])
    axes[0].title.set_text('Original QQ')
    axes[1].title.set_text('Log QQ')
    fig.suptitle(data.name)
    plt.show()
    return log_data


def apply_boxcox(data: np.ndarray|pd.Series, boxcox_lambda=None, alpha: float =0.05) -> [np.ndarray|pd.Series, float]:
    if boxcox_lambda:
        boxcox_data = scipy.stats.boxcox(data, boxcox_lambda, alpha=alpha)
    else:
        boxcox_data, boxcox_lambda, ci = scipy.stats.boxcox(data, alpha=alpha)

    fig, axes = plt.subplots(1, 2)
    sm.ProbPlot(data).qqplot(ax=axes[0])
    sm.ProbPlot(boxcox_data).qqplot(ax=axes[1])
    axes[0].title.set_text('Original QQ')
    axes[1].title.set_text('BoxCox QQ')
    plt.show()
    return boxcox_data, boxcox_lambda


def get_weekday_dummies(dates: np.ndarray|pd.Series) -> np.ndarray:
    weekday_dummies = np.zeros(len(dates))
    weekday_dummies[dates.dayofweek < 5] = 1
    return weekday_dummies


def turn_category_column_to_dummies(data: np.ndarray|pd.Series, prefix_: str = '') -> pd.DataFrame:
    dummies_df = pd.DataFrame(np.zeros((len(data), len(data.unique()))), columns=data.unique())
    for column in dummies_df.columns:
        dummies_df.loc[data == column, column] = 1
    dummies_df.columns = dummies_df.columns.map(lambda x: f'{prefix_}{x}')
    return dummies_df


def exponentiate_forecasts(self) -> None:
    self.train['revenue'] = np.exp(self.train['revenue'])
    for column in ['revenue', 'forecast', 'lower_conf_int', 'upper_conf_int']:
        self.test[column] = np.exp(self.test[column])
    self.test['forecast_error'] = np.abs(self.test['revenue'] - self.test['forecast'])
    self.test['forecast_error_percentage'] = self.test['forecast_error'] / self.test['revenue']
    self.test['conf_int_width'] = self.test['upper_conf_int'] - self.test['lower_conf_int']
    self.test['relative_conf_int_width'] = self.test['conf_int_width'] / self.test['forecast']


def reverse_boxcox(data: np.ndarray|pd.Series, boxcox_lambda: float = 0) -> np.ndarray|pd.Series:
    if boxcox_lambda == 0:
        data = np.exp(data)
    else:
        data = ((data * boxcox_lambda) + 1) ** (1 / boxcox_lambda)
    return data
