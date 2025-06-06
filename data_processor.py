import pandas as pd
import numpy as np
import scipy
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections.abc import Iterable
from typing import Union


def apply_winsorization(data: np.ndarray|pd.Series) -> np.ndarray|pd.Series:
    """
    Applies winsorization to the data, replacing extreme values by mean + three standard errors value
    :param data: data for winsoriization
    :return: winsorized data
    """
    z_score = zscore(data)
    mean = data.mean()
    std = data.std()
    upper_boundary = mean + std * 3
    lower_boundary = -(mean + std * 3)
    data[z_score > 3] = upper_boundary
    data[z_score < -3] = lower_boundary
    return data


def apply_log(data: np.ndarray|pd.Series) -> np.ndarray|pd.Series:
    """
    Applies natural logarithm to the data plus one to account for possible zero values.
    Also plots QQ-plots of the original data and the logged one for visual analysis of normality.
    :param data: data for logging
    :return: data with applied natural logarithm
    """
    log_data = np.log1p(data)
    fig, axes = plt.subplots(1, 2)
    sm.ProbPlot(data).qqplot(ax=axes[0])
    sm.ProbPlot(log_data).qqplot(ax=axes[1])
    axes[0].title.set_text('Original QQ')
    axes[1].title.set_text('Log QQ')
    fig.suptitle(data.name)
    plt.show()
    return log_data


def apply_boxcox(data: np.ndarray|pd.Series, boxcox_lambda=None, alpha: float = 0.05) -> [np.ndarray|pd.Series, float]:
    """
    Applies Box-Cox transformation for data normalization and plots QQ-plots of the original data and the transformed one
    for visual analysis of normality.
    :param data: data for Box-Cox transformation
    :param boxcox_lambda: lambda parameter for transformation. If None, it will be deduced by scipy.stats.boxcox function
    :param alpha: significance level for lambda confidence interval if lambda is None
    :return: transformed data and Box-Cos lambda parameter
    """
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
    """
    Makes nd.array of 0 and 1 values for weekday dummy from the given dates.
    :param dates: dates for dummy
    :return: nd.array of 0 and 1 dummy values
    """
    weekday_dummies = np.zeros(len(dates))
    weekday_dummies[dates.dayofweek < 5] = 1
    return weekday_dummies


def turn_category_column_to_dummies(data: np.ndarray|pd.Series, prefix_: str = '') -> pd.DataFrame:
    """
    Transforms an array of categorical values into a dataframe of respective dummies. Each individual category from
    the array is a column in the returned dataframe. For each index from the original array, 1 is written in the dataframe
    for the row with this index and for the column corresponding to the category stored under this index
    in the original array. Prefix can be added to the names of columns.
    For example,
    If the following column is given with "city_" prefix

    |City    |
    ---------
    |London  |
    |New York|
    |Berlin  |,

    the following dataframe will be returned

    |city_London|city_New York|city_Berlin|
    |1          |0            |0          |
    |0          |1            |0          |
    |0          |0            |1          |

    :param data: array with categories for dummies
    :param prefix_: prefix to be added to the names of columns
    :return: dataframe of categorical dummies
    """
    dummies_df = pd.DataFrame(np.zeros((len(data), len(data.unique()))), columns=data.unique())
    for column in dummies_df.columns:
        dummies_df.loc[data == column, column] = 1
    dummies_df.columns = dummies_df.columns.map(lambda x: f'{prefix_}{x}')
    return dummies_df

def reverse_boxcox(data: np.ndarray|pd.Series, boxcox_lambda: float = 0) -> np.ndarray|pd.Series:
    """
    Transforms Box-Cox transformed data into the original scale.
    :param data: Box-Cox transformed data
    :param boxcox_lambda: lambda parameter that was used for original Box-Cox transformation
    :return: data with the original scale
    """
    if boxcox_lambda == 0:
        data = np.exp(data)
    else:
        data = ((data * boxcox_lambda) + 1) ** (1 / boxcox_lambda)
    return data
