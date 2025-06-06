import numpy.linalg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
import itertools
from typing import Literal


class SARIMAXModel:
    """
    SARIMAX regression model.
    Parameters
    ----------
    freq: str
        frequency of time series data

    Attributes
    ----------
    freq: str
        Frequency of time series data
    best_reg_params: tuple
        Tuple of the best regression parameters (p, d, q, P, D, Q, s, trend) obtained through optimization.
    res: statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Results obtained from statsmodels.tsa.statespace.sarimax.SARIMAX regression
    ljungbox_p_threshold: float
        P-value threshold for the Ljung-Box test during automated optimization of parameters.
    arch_p_threshold: float
        P-value threshold for the ARCH test during automated optimization of parameters.
    jarquebera_p_threshold: float
        P-value threshold for the Jarque-Bera test during automated optimization of parameters.
    coef_p_threshold: float
        Mean p-values threshold for the variable coefficients during automated optimization of parameters.
    forecast_df: pandas.DataFrame
        DataFrame with true values, forecasted values and confidence intervals for out-of-sample forecasts.
    insample_prediction_df: pandas.DataFrame
        DataFrame with true values, predicted values and confidence intervals for in-sample predictions.

    """
    def __init__(self, freq: str):
        self.freq = freq
        self.best_reg_params = None
        self.res = None
        self.ljungbox_p_threshold = None
        self.arch_p_threshold = None
        self.jarquebera_p_threshold = None
        self.coef_p_threshold = None
        self.forecast_df = pd.DataFrame()
        self.insample_prediction_df = pd.DataFrame()

    @staticmethod
    def plot_time_series_differences(endog: pd.Series) -> None:
        """
        Plots time series graph together with its difference and difference of difference
        to visually analyse if differencing is required.
        :param endog: time series data
        :return: None
        """
        fig, axes = plt.subplots(1, 3)
        endog.plot(ax=axes[0], title='Original series')
        endog.diff().plot(ax=axes[1], title='First difference')
        endog.diff().diff().plot(ax=axes[2], title='Second difference')
        plt.show()

    @staticmethod
    def test_serial_correlation(endog: pd.Series, nlags: int) -> None:
        """
        Tests serial correlation using Breusch-Godfrey test and prints LM Statistic, F-Statistic and respective p-values.
        :param endog: time series data
        :param nlags: number of lags for test
        :return: None
        """
        X = sm.add_constant(range(1, len(endog) + 1))
        model = sm.OLS(endog, X).fit()
        bg_test = smd.acorr_breusch_godfrey(model, nlags=nlags)
        print("Breusch-Godfrey Test for Serial Correlation")
        print(f"LM Statistic: {bg_test[0]}")
        print(f"p-value: {bg_test[1]}")
        print(f"F-Statistic: {bg_test[2]}")
        print(f"F p-value: {bg_test[3]}")

    @staticmethod
    def make_white_noise_test(endog: pd.Series, lags: int|list|tuple) -> None:
        """
        Tests time series for white noise using Ljung-Box test and prints LB-statistic and its p-value.
        :param endog: time series data
        :param lags: number of lags for test
        :return: None
        """
        print(f'White noise test: {sm.stats.acorr_ljungbox(endog, lags=[lags], return_df=True)}')

    @staticmethod
    def make_kpss_test(endog: pd.Series, regression: Literal['c', 'ct'], nlags: int) -> None:
        """
        Tests for stationarity using Kwiatkowski–Phillips–Schmidt–Shin test
        and prints KPSS-statistic, p-value, number of lags used and critical values.
        :param endog: time series data
        :param regression: "c" if the null hypothesis is stationarity around a constant,
                           "ct" if the null hypothesis is stationarity around a trend
        :param nlags: number of lags for test
        :return: None
        """
        kpss_stat, p_value, lags, crit_values = kpss(endog, regression=regression, nlags=nlags)
        print('KPSS test')
        print(f"KPSS Statistic: {kpss_stat}")
        print(f"P-Value: {p_value}")
        print(f"Lags Used: {lags}")
        print("Critical Values:", crit_values)

    @staticmethod
    def make_adf_test(endog: pd.Series) -> None:
        """
        Tests for a unit root in the time series using Augmented Dickey-Fuller Test
        and prints test statistic, p-value, number of lags used, number of observations used and critical values.
        :param endog: time series data
        :return: None
        """
        print('ADF Test')
        for adf in [endog, endog.diff()]:
            adft = adfuller(adf.dropna(), autolag='AIC')
            output_df = pd.DataFrame({
                "Values": [adft[0], adft[1], adft[2], adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']],
                "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",
                           "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
            print(output_df)

    @staticmethod
    def test_autocorrelation(endog: pd.Series, autocorr_lags: list|tuple) -> None:
        """
        Tests autocorrelation for given lags and plots Pearson correlation.
        :param endog: time series data
        :param autocorr_lags: list of numbers of lags for test
        :return: None
        """
        for lag in autocorr_lags:
            autocorrelation_lag = endog.autocorr(lag=lag)
            print(f'{lag} lags autocorrelation: ', autocorrelation_lag)

    @staticmethod
    def plot_decompose(endog: pd.Series, n_seasonal_lags: int) -> None:
        """
        Plots seasonal-trend decomposition.
        :param endog: time series data
        :param n_seasonal_lags: number of seasonal lags for decomposition
        :return: None
        """
        decompose = STL(endog, period=n_seasonal_lags).fit()
        decompose.plot()
        plt.show()

    @staticmethod
    def plot_acf_pacf(endog: pd.Series, n_seasonal_lags: int) -> None:
        """
        Plots ACF and PACF for original data and seasonally differenced with the given number of lags.
        :param endog: time series data
        :param n_seasonal_lags: number of seasonal lags for differencing
        :return: None
        """
        fig, axes = plt.subplots(2, 2)
        sm.graphics.tsa.plot_acf(endog, ax=axes[0, 0], title='ACF original')
        sm.graphics.tsa.plot_pacf(endog, ax=axes[0, 1], title='PACF original')
        sm.graphics.tsa.plot_acf(endog.diff(periods=n_seasonal_lags).dropna(), ax=axes[1, 0],
                                 title='ACF seasonally differenced')
        sm.graphics.tsa.plot_pacf(endog.diff(periods=n_seasonal_lags).dropna(), ax=axes[1, 1],
                                  title='PACF seasonally differenced')
        plt.show()

    @staticmethod
    def train_test_split_reg_df(data: pd.DataFrame, test_length: int) -> (pd.DataFrame, pd.DataFrame):
        """
        Splits original data into train and test set based on the number of periods for the test set.
        :param data: time series data
        :param test_length: number of periods to be included in the test set
        :return: two dataframes for train and test parts of the data
        """
        train_df = data[:-test_length]
        test_df = data[-test_length:]
        return train_df, test_df

    def optimize_reg_params(self, endog_train: pd.Series, endog_test: pd.Series, forecast_length: int,
                            p_params: tuple, d_params: tuple, q_params: tuple,
                            season_p_params: tuple, season_d_params: tuple, season_q_params: tuple, season_params: tuple,
                            trend_params: tuple, exog_train: pd.DataFrame, exog_test: pd.DataFrame,
                            ljungbox_p_threshold: float = 0, arch_p_threshold: float = 0,
                            jarquebera_p_thershold: float = 0, coef_p_threshold: float = 1,
                            relative_conf_int_width_threshold: float|int = np.inf, optimize_param: str = 'AIC') -> None:
        """
        Iteratively searches for best regression parameters of all possible combinations from the given options.
        Optimization is performed based on either AIC or RMSE values. Optionally, it is possible to set thresholds
        for p-values for:
            1) white noise test
            2) heteroskedasticity test
            3) residuals normality test
            4) mean coefficients p-values
            5) relative confidence interval width for in-sample prediction and out-of-sample forecast
        :param endog_train: time series data for training purpose
        :param endog_test: time series data for testing purpose
        :param forecast_length: number of periods for out-of-sample forecast when comparing with endog_test
        :param p_params: list of options for autoregressive part
        :param d_params: list of options for integration (differencing)
        :param q_params: list of options for moving-average part
        :param season_p_params: list of options for seasonal autoregressive part
        :param season_d_params: list of options for seasonal integration (differencing)
        :param season_q_params: list of options for seasonal moving-average part
        :param season_params: list of options for seasonal periodicity
        :param trend_params: list of option for trend parameters
                             ("n" for no trend, "c" for constant trend, "t" for linear trend and "ct" for combined trend)
        :param exog_train: data for exogenous variables for training purpose
        :param exog_test: data for exogenous variables for testing purpose
        :param ljungbox_p_threshold: p-value threshold for Ljung-Box white noise test
        :param arch_p_threshold: p-value threshold for ARCH test
        :param jarquebera_p_thershold: p-value threshold for Jarque-Bera test
        :param coef_p_threshold: p-value threshold for coefficients
        :param relative_conf_int_width_threshold: relative confidence interval width threshold
        :param optimize_param: criteria for optimization ("AIC" or "RMSE")
        :return: None, results are stored as class attributes
        """
        lowest_aic = None
        lowest_rmse = None
        self.ljungbox_p_threshold = ljungbox_p_threshold
        self.arch_p_threshold = arch_p_threshold
        self.jarquebera_p_threshold = jarquebera_p_thershold
        self.coef_p_threshold = coef_p_threshold
        for params_tuple in itertools.product(p_params, d_params, q_params,
                                              season_p_params, season_d_params, season_q_params, season_params,
                                              trend_params):
            if params_tuple[0] == 0 and params_tuple[2] == 0:
                continue
            if params_tuple[3] == 0 and params_tuple[4] == 0 and season_params != 0:
                continue
            try:
                self.run_single_regression(endog_train,
                                           params_tuple[0], params_tuple[1], params_tuple[2],
                                           params_tuple[3], params_tuple[4], params_tuple[5], params_tuple[6],
                                           params_tuple[7],
                                           exog=exog_train)
                if self.test_results():
                    continue
                self.make_forecast(forecast_length, exog_test)
                self.forecast_df.index = endog_test.index
                self.calculate_forecast_error(endog_test)
                self.make_insample_prediction(1, len(endog_train) - 1, seasonality=params_tuple[6])
                self.calculate_insample_prediction_error(endog_train)
                if self.forecast_df['relative_conf_int_width'].mean() > relative_conf_int_width_threshold:
                    continue
                if self.insample_prediction_df['relative_conf_int_width'].mean() > relative_conf_int_width_threshold:
                    continue
                if self.forecast_df['relative_conf_int_width'].mean() == np.inf:
                    continue
                if self.insample_prediction_df['relative_conf_int_width'].mean() == np.inf:
                    continue

                if optimize_param == 'AIC':
                    if lowest_aic is None:
                        lowest_aic = self.res.aic
                        self.best_reg_params = params_tuple
                    else:
                        if lowest_aic > self.res.aic:
                            lowest_aic = self.res.aic
                            self.best_reg_params = params_tuple
                elif optimize_param == 'RMSE':
                    reg_rmse = rmse(self.forecast_df['endog_true'], self.forecast_df['predicted_mean'])
                    if lowest_rmse is None:
                        lowest_rmse = reg_rmse
                        self.best_reg_params = params_tuple
                    else:
                        if lowest_rmse > reg_rmse:
                            lowest_rmse = reg_rmse
                            self.best_reg_params = params_tuple
                else:
                    raise Exception('Unknown parameter for optimization')
            except numpy.linalg.LinAlgError:
                continue
            except ValueError as e:
                print(e)
                print(f'Error was caused by parameters: {params_tuple}')
                continue
            except statsmodels.tools.sm_exceptions.MissingDataError:
                continue
        if self.best_reg_params is None:
            raise Exception('No model was found under given p-values constraints')
        self.run_single_regression(endog_train,
                                   self.best_reg_params[0], self.best_reg_params[1],
                                   self.best_reg_params[2], self.best_reg_params[3],
                                   self.best_reg_params[4], self.best_reg_params[5],
                                   self.best_reg_params[6], self.best_reg_params[7],
                                   exog=exog_train
                                   )
        self.make_forecast(forecast_length, exog_test)
        self.forecast_df.index = endog_test.index
        self.calculate_forecast_error(endog_test)
        self.make_insample_prediction(1, len(endog_train) - 1, seasonality=self.best_reg_params[6])
        self.calculate_insample_prediction_error(endog_train)
        reg_rmse = rmse(self.forecast_df['endog_true'], self.forecast_df['predicted_mean'])
        print(f'Best parameters: {self.best_reg_params}, AIC: {self.res.aic}, RMSE: {reg_rmse}')

    def run_single_regression(self, endog: pd.Series, p: int, d: int, q: int,
                              season_p: int, season_d: int, season_q: int, s: int, trend: str | list,
                              exog: pd.DataFrame|pd.Series = None, maxiter: int = 1000) -> None:
        """
        Runs an ARIMA, SARIMA or SARIMAX regression with given parameters.
        Optionally,maximum number of iterations for solving can be changed
        :param endog: time series data
        :param p: autoregressive part parameter
        :param d: integration (differencing) parameter
        :param q: moving-average part parameter
        :param season_p: seasonal autoregressive part parameter
        :param season_d: seasonal integration (differencing) parameter
        :param season_q: seasonal moving-average part parameter
        :param s: seasonal periodicity parameter
        :param trend: trend parameter
                      ("n" for no trend, "c" for constant trend, "t" for linear trend and "ct" for combined trend)
        :param exog: data for exogenous variables
        :param maxiter: maximum number of iterations for solving
        :return: None, results are stored as class attribute
        """
        model = SARIMAX(endog, exog=exog,
                        order=(p, d, q),
                        seasonal_order=(season_p, season_d, season_q, s), trend=trend, freq=self.freq)
        self.res = model.fit(maxiter=maxiter)

    def test_results(self) -> bool:
        """
        Tests regression results using coefficients z-values, Ljung-Box test, ARCH test, Jarque-Bera test
        and coefficients p-values.
        :return: boolean value, whether any of the test results are inadequate
        """
        zvalues_inf = np.inf in tuple(self.res.zvalues) or -np.inf in tuple(self.res.zvalues)
        lb_test = self.make_ljungbox_test()
        arch_test =  self.make_arch_test()
        jb_test = self.make_jarquebera_test()
        pvalues = self.res.pvalues.mean() > self.coef_p_threshold
        pvalues_na = self.res.pvalues.isna().any()
        return any((zvalues_inf, lb_test, arch_test, jb_test, pvalues, pvalues_na))

    def make_ljungbox_test(self) -> bool:
        """
        Performs Ljung-Box test for white noise on the regression results.
        :return:  boolean value, if the test results are inadequate
        """
        residuals = self.res.resid
        lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
        return lb_test['lb_pvalue'].iloc[0] < self.ljungbox_p_threshold

    def make_arch_test(self) -> bool:
        """
        Performs ARCH test for heteroskedasticity on the regression results.
        :return: boolean value, if the test results are inadequate
        """
        test_result = het_arch(self.res.resid)
        lm_stat, lm_pval, f_stat, f_pval = test_result
        return lm_pval < self.arch_p_threshold or f_pval < self.arch_p_threshold

    def make_jarquebera_test(self) -> bool:
        """
        Performs Jarque-Bera test for normality of residuals from the regression results.
        :return:boolean value, if the test results are inadequate
        """
        jb_test = self.res.test_normality(method="jarquebera")
        return jb_test[0][1] < self.jarquebera_p_threshold

    def make_forecast(self, forecast_length: int, exog: pd.DataFrame, conf_int_alpha: float = 0.05) -> None:
        """
        Makes forecast from the calculated regression for a specific number of periods.
        :param forecast_length: number of periods for forecast
        :param exog: data for exogenous variables for forecast periods
        :param conf_int_alpha: significance level for confidence interval
        :return: None, results are saved as class attributes
        """
        forecast = self.res.get_forecast(steps=forecast_length, exog=exog)
        self.forecast_df = pd.DataFrame({'predicted_mean': forecast.predicted_mean.values,
                                         'lower_conf_int': forecast.conf_int(alpha=conf_int_alpha).values[:, 0],
                                         'upper_conf_int': forecast.conf_int(alpha=conf_int_alpha).values[:, 1]})

    def calculate_forecast_error(self, endog_true: pd.Series) -> None:
        """
        Calculates forecast error as difference between true values and forecasted ones, and confidence interval width.
        For both also calculates relative values.
        :param endog_true: true values for the forecasted periods
        :return: None, results are saved as class attributes
        """
        self.forecast_df['endog_true'] = endog_true
        self.forecast_df['forecast_error'] = np.abs(self.forecast_df['endog_true'] - self.forecast_df['predicted_mean'])
        self.forecast_df['forecast_error_percentage'] = self.forecast_df['forecast_error'] / self.forecast_df['endog_true']
        self.forecast_df['conf_int_width'] = self.forecast_df['upper_conf_int'] - self.forecast_df['lower_conf_int']
        self.forecast_df['relative_conf_int_width'] = self.forecast_df['conf_int_width'] / self.forecast_df['predicted_mean']

    def make_insample_prediction(self, start: int, end: int, conf_int_alpha: float = 0.05, seasonality: int = 0) -> None:
        """
        Makes insample prediction
        :param start: first observation index from which to start prediction
        :param end: last observation index
        :param conf_int_alpha: significance level for confidence interval
        :param seasonality: seasonality of the time series. If start is lower than seasonality,
        seasonality number of periods will be cut off from the beginning of the prediction.
        :return: None, results are saved as class attributes
        """
        insample_prediction = self.res.get_prediction(start, end)
        self.insample_prediction_df = pd.DataFrame(
            {'predicted_mean': insample_prediction.predicted_mean,
             'lower_conf_int': insample_prediction.conf_int(alpha=conf_int_alpha).values[:, 0],
             'upper_conf_int': insample_prediction.conf_int(alpha=conf_int_alpha).values[:, 1]})
        if start < seasonality:
            self.insample_prediction_df.iloc[:seasonality, :] = np.nan

    def calculate_insample_prediction_error(self, endog_true: pd.Series) -> None:
        """
        Calculates imsaple prediction error as difference between true values and predicted ones, and confidence interval width.
        For both also calculates relative values.
        :param endog_true: true values for the predicted periods
        :return: None, results are saved as class attributes
        """
        self.insample_prediction_df['endog_true'] = endog_true
        self.insample_prediction_df['prediction_error'] = (
            np.abs(self.insample_prediction_df['endog_true'] - self.insample_prediction_df['predicted_mean']))
        self.insample_prediction_df['prediction_error_percentage'] = (
                self.insample_prediction_df['prediction_error'] / self.insample_prediction_df['endog_true'])
        self.insample_prediction_df['conf_int_width'] = (
                self.insample_prediction_df['upper_conf_int'] - self.insample_prediction_df['lower_conf_int'])
        self.insample_prediction_df['relative_conf_int_width'] = (
                self.insample_prediction_df['conf_int_width'] / self.insample_prediction_df['predicted_mean'])

    def print_results(self):
        """
        Prints regression results summary, forecast and in-sample prediction errors and cofidence width
        and also plots true values and predicted ones.
        :return: None
        """
        print(self.res.summary())
        print(f'Mean forecast error: {self.forecast_df['forecast_error'].mean()}')
        print(f'Mean forecast error percentage: {self.forecast_df["forecast_error_percentage"].mean()}')
        print(f'Mean forecast relative confidence interval width: {self.forecast_df['relative_conf_int_width'].mean()}')
        print(f'Mean in-sample prediction error: {self.insample_prediction_df['prediction_error'].mean()}')
        print(f'Mean in-sample prediction error percentage: {
            self.insample_prediction_df["prediction_error_percentage"].mean()}')
        print(f'Mean in-sample prediction relative confidence interval width: {
            self.insample_prediction_df['relative_conf_int_width'].mean()}')

        fig, axes = plt.subplots(1, 1)
        axes.plot(self.insample_prediction_df['endog_true'], color='black', label='Train data')
        axes.plot(self.forecast_df.index, self.forecast_df['predicted_mean'], color='green', label='Forecast')
        axes.plot(self.forecast_df.index, self.forecast_df['endog_true'], color='red', label='Test data')
        axes.plot(self.insample_prediction_df['predicted_mean'], color='blue', label='In-sample prediction')
        plt.fill_between(self.forecast_df.index,
                         self.forecast_df['lower_conf_int'], self.forecast_df['upper_conf_int'],
                         alpha=0.05, color='b')
        plt.fill_between(self.insample_prediction_df.index,
                         self.insample_prediction_df['lower_conf_int'], self.insample_prediction_df['upper_conf_int'],
                         alpha=0.05, color='purple')
        axes.legend(loc='best')
        plt.show()
