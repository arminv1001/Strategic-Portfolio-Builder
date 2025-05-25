import datetime as dt
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import yaml
import json
import os
import pandas_datareader.data as web
from datetime import timedelta
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# I. Data acquisition & preprocessing
# ---------------------------------------------------------------------------


def download_prices(
    tickers: List[str],
    start: str = "2010-01-01",
    end: str = None,
    interval: str = "1mo",
) -> pd.DataFrame:
    """Download *adjusted* close prices and forward‑fill any missing values.

    Args:
        tickers (List[str]): List of stock tickers to download.
        start (str, optional): Start date for the data. Defaults to "2010-01-01".
        end (str, optional): End date for the data. Defaults to None.
        interval (str, optional): Data interval. Defaults to "1mo".

    Returns:
        pd.DataFrame: DataFrame containing the downloaded price data.
    """
    if end is None:
        end = dt.date.today().isoformat()
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=True,
    ).dropna()
    return data


def get_risk_free_rate_date(target_date: pd.Timestamp, config: dict):
    """Fetch the risk-free rate for a specific date.

    Args:
        target_date (pd.Timestamp): The date for which to retrieve the risk-free rate.
        config (dict): Configuration dictionary containing risk-free rate settings.

    Raises:
        ValueError: If no valid risk-free rate is found for the specified date.

    Returns:
        float: The risk-free rate as a decimal (e.g. 0.015 for 1.5%).
    """
    MANUAL = 0
    ONLINE = 1

    # Check if the risk-free rate is set to manual or online
    if (
        config["risk_free_rate"][ONLINE]["online_risk_free_rate"] is False
        and config["risk_free_rate"][MANUAL]["manual_risk_free_rate"] is True
    ):
        # If the online risk-free rate is turned off, return a default value
        return config["risk_free_rate"][MANUAL]["value_risk_free_rate"]

    else:
        series = config["risk_free_rate"][ONLINE]["online_risk_free_rate"]["series"]
        # Fetch a window of data ending on the target date
        start = target_date - timedelta(days=7)
        df = web.DataReader(series, "fred", start, target_date)

        # Isolate the series and drop NaN
        yields = df[series].dropna()
        # Filter to dates on or before the target date
        yields = yields[yields.index <= target_date]
        if yields.empty:
            raise ValueError(
                f"No valid {series} data on or before {target_date.date()}"
            )

        # Take the last available yield and convert to decimal
        latest_pct = yields.iloc[-1]
        return round(latest_pct / 100.0, 4)  # Convert to decimal and round to 4 decimal places



def get_monthly_rf_dgs3mo(start, end):
    """
    Fetches the 3-month Treasury constant maturity yield (DGS3MO) from FRED,
    resamples to month-end, and converts annual yield (%) to a monthly rate.

    Parameters
    ----------
    start : str or datetime-like
        Start date (e.g. "2015-01-01").
    end : str or datetime-like
        End date   (e.g. "2025-05-31").

    Returns
    -------
    pd.Series
        Index = month-end dates,
        Values = monthly risk-free rate (decimal, e.g. 0.0023 for 0.23%).
    """
    # Pull daily series of the annualized yield (%)
    rf_daily = web.DataReader("DGS3MO", "fred", start, end)

    # Resample to get the last observation of each calendar month
    rf_monthly_annual_pct = rf_daily["DGS3MO"].resample("ME").last()

    # Convert percent to decimal
    rf_monthly_annual = rf_monthly_annual_pct / 100.0

    # Convert annual yield to equivalent monthly rate
    rf_monthly_rate = (1 + rf_monthly_annual) ** (1 / 12) - 1

    return rf_monthly_rate


def price_to_returns(prices: pd.DataFrame, log: bool) -> pd.DataFrame:
    """Convert price series to log returns or simple returns.

    Args:
        prices (pd.DataFrame): DataFrame containing price series.
        log (bool): If True, compute log returns; if False, compute simple returns.

    Returns:
        pd.DataFrame: DataFrame containing return series.
    """
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return prices.pct_change().dropna()


def save_stats(stats: dict, project_name: str):
    """Save summary statistics to a JSON file."""
    os.makedirs(f"output_data/{project_name}", exist_ok=True)
    with open(f"output_data/{project_name}/stats_asset_summary.json", "w") as f:
        json.dump(stats, f, indent=4)

def prep_data(config: dict, tickers_data: pd.DataFrame, benchmark_data: pd.DataFrame):
    """Prepare the data for analysis by splitting into in-sample and out-of-sample sets.

    Args:
        config (dict): Configuration dictionary containing dataset split settings.
        tickers_data (pd.DataFrame): DataFrame containing ticker price data.
        benchmark_data (pd.DataFrame): DataFrame containing benchmark price data.

    Returns:
        tuple: A tuple containing the in-sample and out-of-sample data for both tickers and benchmark.
    """
    if "dataset_split" in config:
        split = int(len(tickers_data) * config["dataset_split"])
        tickers_data_insample = tickers_data.iloc[:split]
        tickers_data_out_sample = tickers_data.iloc[split:]
        benchmark_data_insample = benchmark_data.iloc[:split]
        benchmark_data_out_sample = benchmark_data.iloc[split:]
    else:
        tickers_data_insample = tickers_data
        tickers_data_out_sample = None
        benchmark_data_insample = benchmark_data
        benchmark_data_out_sample = None

  

    rets_insample = price_to_returns(tickers_data_insample["Adj Close"], log=config.get("log_returns", False))

    rets_bench_insample = price_to_returns(benchmark_data_insample["Adj Close"], log=config.get("log_returns", False))

    stats_table = summarise_returns(rets_insample, freq=config["stats_frequency"])
    return tickers_data_insample, tickers_data_out_sample, benchmark_data_insample, benchmark_data_out_sample, rets_insample, rets_bench_insample, stats_table


# ---------------------------------------------------------------------------
# II. Descriptive statistics
# ---------------------------------------------------------------------------


def summarise_returns(ret: pd.DataFrame, freq: int = 12) -> dict:
    """Compute summary statistics for a DataFrame of returns.

    Args:
        ret (pd.DataFrame): DataFrame containing return series.
        freq (int, optional): Frequency for annualization. Defaults to 12.

    Returns:
        dict: Dictionary containing summary statistics for each asset.
    """
    summary = {}
    for ticker in ret.columns:
        jb_stat, jb_pval = stats.jarque_bera(ret[ticker])
        summary[f"{ticker} mu"] = ret[ticker].mean() * freq
        summary[f"{ticker} std"] = ret[ticker].std() * freq**0.5
        summary[f"{ticker} JB Stat"] = jb_stat
        summary[f"{ticker} JB p-val"] = jb_pval
    return summary


# ---------------------------------------------------------------------------
# III. Mean‑variance optimisation helpers
# ---------------------------------------------------------------------------


def uncon_mean_var_frontier(mu, sigma, n=500):
    """Compute the unconstrained minimum-variance frontier.

    Args:
        mu (np.ndarray): Expected returns of assets.
        sigma (np.ndarray): Covariance matrix of asset returns.
        n (int, optional): Number of points on the frontier. Defaults to 500.

    Returns:
        tuple: (weights, expected_returns, volatilities)
    """
    sigma_inv = np.linalg.inv(sigma) 
    fully_invest_vector = np.ones((mu.shape[0], 1))  # full invest vector
    X = (mu.T @ sigma_inv @ mu).item()
    Y = (mu.T @ sigma_inv @ fully_invest_vector).item()
    Z = (fully_invest_vector.T @ sigma_inv @ fully_invest_vector).item()

    # Calculate the weights
    D = X * Z - Y**2
    g = (1 / D) * (X * sigma_inv @ fully_invest_vector - Y * sigma_inv @ mu)
    h = (1 / D) * (Z * sigma_inv @ mu - Y * sigma_inv @ fully_invest_vector)

    incr = (0.6 - min(mu)) / (n - 1)

    w_MV = np.zeros((n, mu.shape[0]))
    mu_MV = np.zeros((n, 1))
    sigma_MV = np.zeros((n, 1))

    for i in range(n):
        mu_i = i * incr
        w_i = g + h * mu_i
        w_MV[i, :] = w_i.T
        mu_MV[i] = mu.T @ w_i
        sigma_MV[i] = (w_i.T @ sigma @ w_i).item() ** 0.5

    return w_MV, mu_MV, sigma_MV


def constrained_mean_var_frontier(mu, sigma, config: dict, n=500):
    """Compute the constrained minimum-variance frontier.

    Args:
        mu (np.ndarray): Expected returns of assets.
        sigma (np.ndarray): Covariance matrix of asset returns.
        config (dict): Configuration dictionary containing constraints.
        n (int, optional): Number of points on the frontier. Defaults to 500.

    Returns:
        tuple: (weights, expected_returns, volatilities)
    """
    m = mu.shape[0]
    # Grid of target returns
    mu_min, mu_max = mu.min(), mu.max()
    mu_incr = (mu_max - mu_min) / (n - 1)
    mu_target = mu_min

    # Portfolio variance function
    def port_var(w):
        """Calculate portfolio variance."""
        return (w @ sigma @ w).item()  # returns a scalar

    # Equality constraints
    def eq_target_return(w):
        """Equality constraint for target return."""
        return w @ mu - mu_target

    def eq_full_investment(w):
        """Equality constraint for full investment."""
        return np.sum(w) - 1.0

    # Constraints
    constraints = [
        {"type": "eq", "fun": eq_full_investment},
        {"type": "eq", "fun": eq_target_return},
    ]

    # Bounds
    bounds = [
        (
            int(config["conditioning"]["allow_shorting"]),
            config["conditioning"]["max_allocation_threshold"],
        )
    ] * m
    # Storage
    w_MVF_c = np.zeros((n, m))
    mu_MVF = np.zeros((n, 1))
    sigma_MVF = np.zeros((n, 1))
    w0 = np.ones(m) / m  # Initial guess: equally weighted
    # Loop over target returns
    for i in range(n):
        # Minimise portfolio variance for this target return
        w_i = minimize(port_var, w0, constraints=constraints, bounds=bounds)
        w_MVF_c[i, :] = w_i.x
        mu_MVF[i] = w_MVF_c[i, :] @ mu
        sigma_MVF[i] = (w_MVF_c[i, :] @ sigma @ w_MVF_c[i, :].T).item() ** 0.5

        mu_target += mu_incr  # Increment target return

    return w_MVF_c, mu_MVF, sigma_MVF


def plot_mean_variance_frontier(sigma_MV, mu_MV, sigma, mu, plot_picture=True, TP_index=None, rf=None):
    """Plot the mean-variance frontier.

    Args:
        sigma_MV (np.ndarray): Volatilities of the minimum-variance portfolios.
        mu_MV (np.ndarray): Expected returns of the minimum-variance portfolios.
        sigma (np.ndarray): Covariance matrix of asset returns.
        mu (np.ndarray): Expected returns of assets.
        plot_picture (bool, optional): Whether to plot the picture. Defaults to True.
        TP_index (int, optional): Index of the tangency portfolio. Defaults to None.
        rf (float, optional): Risk-free rate. Defaults to None.
    """
    plt.figure(1, figsize=(10, 6))
    plt.plot(sigma_MV, mu_MV, label="Mean-Variance Frontier")
    plt.scatter(np.diag(sigma) ** 0.5, mu[:, 0], marker="o", label="Assets")

    if TP_index is not None and rf is not None:
        # Plot the tangency portfolioƒ
        plt.scatter([0], [rf], s=20, color="red", label="Risk-free rate")
        plt.scatter(
            sigma_MV[TP_index, 0],
            mu_MV[TP_index, 0],
            marker="*",
            s=150,
            label="Tangency Portfolio",
        )
        sr_tp = (mu_MV[TP_index, 0] - rf) / sigma_MV[TP_index, 0]
        x_tp = np.linspace(0, np.max(sigma_MV))
        y_tp = rf + sr_tp * x_tp
        plt.plot(x_tp, y_tp, color="gray", label="Capital Allocation Line (CAL)")

    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return (µ)")
    plt.title("Mean-Variance Frontier")
    plt.legend()
    plt.grid()
    if plot_picture == False:
        plt.savefig(f"output_data/{config['project_name']}/mean_variance_frontier.png")
    else:
        plt.show()


def tangency_portfolio(
    mu_MV: np.ndarray, sigma_MV: np.ndarray, rf: float
) -> np.ndarray:
    """Compute the weights of the tangency portfolio.

    Args:
        mu_MV (np.ndarray): Expected returns of the assets.
        sigma_MV (np.ndarray): Volatilities of the assets.
        rf (float): Risk-free rate.

    Returns:
        np.ndarray: Weights of the tangency portfolio.
    """
    SR_M = (mu_MV - rf) / sigma_MV
    TP_index = np.argmax(SR_M)
    return TP_index


def jb_normality_test(stats_info: dict,tickers:list, threshold: float) -> list:
    """Run Jarque–Bera normality test on each asset's returns.

    Args:
        stats (dict): Dictionary containing summary statistics.
        threshold (float): p-value threshold for normality.

    Returns:
        list: List of assets that are not normally distributed.
    """
    non_normal = []
    for ticker in tickers:
        if (stats_info[f"{ticker} JB p-val"] > threshold):
            non_normal.append(ticker.split(" ")[0])
    print(f"This is the list of non-normal assets: {non_normal}")
    return non_normal


def calc_mu(stats_info: dict, tickers: list) -> np.ndarray:
    """Calculate the expected returns (mu) from the summary statistics.

    Args:
        stats_info (dict): Dictionary containing summary statistics.
        tickers (list): List of asset tickers.

    Returns:
        np.ndarray: Array of expected returns.
    """
    # Collect all mu
    mu = []
    for t in tickers:
        mu.append(float(stats_info[f"{t} mu"]))
    # Convert to numpy array
    mu = np.array(mu).reshape(-1, 1)
    return mu


def calc_sigma(stats_info: dict, corr_matrix: np.ndarray) -> np.ndarray:
    """Calculate the covariance matrix (sigma) from the summary statistics.

    Args:
        stats_info (dict): Dictionary containing summary statistics.
        corr_matrix (np.ndarray): Correlation matrix of asset returns.

    Returns:
        np.ndarray: Covariance matrix.
    """
    vola = []
    for t in config["tickers"]:
        vola.append(float(stats_info[f"{t} std"]))
    vola = np.array(vola)
    sigma_diag = np.sqrt(np.diag(vola))
    sigma = sigma_diag @ corr_matrix @ sigma_diag
    return sigma


# ---------------------------------------------------------------------------
# IV. CAPM
# ---------------------------------------------------------------------------


def capm_decomposition(
    portfolio_returns: pd.Series, market_returns: pd.Series
) -> pd.DataFrame:
    """Decompose portfolio returns into systematic and idiosyncratic components.

    Args:
        portfolio_returns (pd.Series): Portfolio returns.
        market_returns (pd.Series): Market returns (e.g., SPY).

    Returns:
        pd.DataFrame: DataFrame with portfolio returns, market returns, and CAPM components.
    """

    rf_monthly = get_monthly_rf_dgs3mo(
        portfolio_returns.index.min(), portfolio_returns.index.max()
    )
    # Align the risk-free rate index with the portfolio returns
    rf_monthly = rf_monthly.reindex(portfolio_returns.index, method="ffill")

    xr_portfolio = portfolio_returns - rf_monthly.shift(1)  # Excess portfolio returns
    xr_market = market_returns - rf_monthly.shift(1)  # Excess market returns
    xr_portfolio = xr_portfolio.dropna()
    xr_market = xr_market.dropna()

    # Regression
    X = sm.add_constant(xr_market.values)  # Add constant for alpha
    model = sm.OLS(xr_portfolio, X).fit()
    alpha_p, beta_p = model.params[0], model.params[1]

    # Expected components
    exp_mkt_exc = xr_market.mean()
    sys_return = beta_p * exp_mkt_exc
    idio_return = alpha_p
    total_return = xr_portfolio.mean()

    return {
        "Portfolio Beta": float(beta_p),
        "Portfolio Alpha": float(alpha_p),
        "Expected Systematic Return": float(sys_return),
        "Expected Idiosyncratic Return": float(idio_return),
        "Expected Total Excess Return": float(total_return),
    }


# ---------------------------------------------------------------------------
# V. Backtesting & portfolio simulation
# ---------------------------------------------------------------------------


def compute_backtest_stats(returns: pd.Series, rf_annual: float = 0.0, freq: int = 12):
    """Compute key backtest statistics from a periodic returns series.

    Args:
        returns (pd.Series): Periodic returns series.
        rf_annual (float, optional): Annual risk-free rate. Defaults to 0.0.
        freq (int, optional): Frequency of returns (e.g., 12 for monthly). Defaults to 12.

    Returns:
        dict: Dictionary containing key backtest statistics.
    """
    # Total cumulative return
    total_return = (1 + returns).prod() - 1
    # CAGR
    days = (returns.index[-1] - returns.index[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    # Annualized volatility
    vol = returns.std(ddof=1) * np.sqrt(freq)
    # Sharpe ratio
    sharpe = (cagr - rf_annual) / vol if vol != 0 else np.nan
    
    # Win/loss stats
    win_rate = (returns > 0).sum() / len(returns)
    avg_gain = returns[returns > 0].mean() if (returns > 0).any() else np.nan
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else np.nan
  
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Win Rate": win_rate,
        "Average Gain": avg_gain,
        "Average Loss": avg_loss,
    }


def run_portfolio(
    future_returns: pd.Series,
    config: dict,
    portfolio_stats: dict,
    rf: float,
    plot_picture=True,
):
    """Run portfolio simulation and analyze performance.

    Args:
        future_returns (pd.Series): Future returns of the portfolio.
        config (dict): Configuration dictionary.
        portfolio_stats (dict): Portfolio statistics.
        rf (float): Risk-free rate.
        plot_picture (bool, optional): Whether to plot the performance picture. Defaults to True.

    Returns:
        pd.Series: Simulated portfolio values over time.
        dict: Dictionary containing backtest statistics.
    """

    # Configuration
    start_capital = config.get("start_capital", 1_000_000)
    A = config["risk_aversion"]

    # 1) Determine weight y in risky portfolio
    y = (portfolio_stats["expected_return"] - rf) / (
        A * portfolio_stats["expected_volatility"]
    )

    if y > 1:
        y = 1  # Cap at 100% in risky assets
    elif y < 0:
        y = 0 # Cap at 0% in risky assets

    # Calculate portfolio values
    initial_investment = start_capital * y
    cum_return = (1 + future_returns).cumprod()
    risky_value = initial_investment * cum_return
    riskfree_value = (1 - y) * start_capital * (portfolio_stats["risk_free_rate"] + 1)
    # combine: value at each time
    portfolio_value = risky_value + riskfree_value
    portfolio_value.index = future_returns.index

    # Compute drawdown series
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1

    # Compute stats
    stats = compute_backtest_stats(
        returns=future_returns,
        rf_annual=rf,
        freq=12,
    )
    stats["Weight in risky portfolio (y)"] = y
    stats["Max Drawdown"] = drawdown.min()
    # Plot portfolio value and drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(portfolio_value, label="Portfolio Value")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax2.plot(drawdown, label="Drawdown", color="tab:red")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    if plot_picture:
        plt.show()
    else:
        plt.savefig(f"output_data/{config['project_name']}/portfolio_performance.png")

    return portfolio_value, stats


def capm_analysis(capm_stats: dict, config: dict):
    """Perform CAPM analysis on the portfolio.

    Args:
        capm_stats (dict): Dictionary containing CAPM statistics.
        config (dict): Configuration dictionary.

    Returns:
        float: Portfolio health score.
    """
    port_health = 0
    if capm_stats["Portfolio Beta"] > config["capm_health"]["beta_threshold"]:
        port_health += 0.5
    if capm_stats["Portfolio Alpha"] < config["capm_health"]["alpha_threshold"]:
        port_health += 0.5

    return port_health


# ---------------------------------------------------------------------------
# VII. Main workflow
# ---------------------------------------------------------------------------


def run_analysis(config: dict):
    """Run the analysis workflow for the portfolio.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: DataFrame containing the analysis results.
    """

    PLOT_PICTURE = config.get("plot_picture", True)

    tickers_data = download_prices(
        config["tickers"],
        config["start_date"],
        config.get("end_date", None),
        config.get("data_frequency", "1mo"),
    )
    benchmark_ticker = config.get("benchmark_ticker", "SPY")
    benchmark_data = download_prices(
        [benchmark_ticker],
        config["start_date"],
        config.get("end_date", None),
        config.get("data_frequency", "1mo"),
    )
    tickers_data_insample, tickers_data_out_sample, benchmark_data_insample, benchmark_data_out_sample, rets_insample, rets_bench_insample, stats_table = prep_data(config, tickers_data, benchmark_data)
    jb_normality_test_result = jb_normality_test(stats_table, config["tickers"],config["filter"]["jarque_bera_threshold"])
    # If the Jarque-Bera test results in dropping tickers, update the config and data
    if len(jb_normality_test_result) != len(config["tickers"]) and len(jb_normality_test_result) > 1:
        tickers_to_drop = set(config["tickers"]) - set(jb_normality_test_result)
        print(f"Dropping tickers {tickers_to_drop} due to Jarque-Bera test.")
        config["tickers"] = [t for t in config["tickers"] if t not in tickers_to_drop]
        tickers_data = download_prices( config["tickers"], config["start_date"], config.get("end_date", None), config.get("data_frequency", "1mo"))
        tickers_data_insample, tickers_data_out_sample, benchmark_data_insample, benchmark_data_out_sample, rets_insample, rets_bench_insample, stats_table = prep_data(config, tickers_data, benchmark_data)
    elif len(jb_normality_test_result) <= 1:
        print("No valid tickers left after Jarque-Bera test. Exiting analysis.")
        return True

    del tickers_data, benchmark_data
    # Extract mu from stats
    mu = calc_mu(stats_table, config["tickers"])
    # Calculate sigma
    sigma = calc_sigma(stats_table, rets_insample.corr().to_numpy())

    if config["conditioning"]["activate"]:
        w_MV, mu_MV, sigma_MV = constrained_mean_var_frontier(mu, sigma, config)
    else:
        w_MV, mu_MV, sigma_MV = uncon_mean_var_frontier(mu, sigma)
    # plot_mean_variance_frontier(sigma_MV, mu_MV, sigma, mu)

    # Calculate annualised risk-free rate
    rf = get_risk_free_rate_date(tickers_data_insample.index[-1], config)
    # Tangency & optimal complete portfolio
    TP_index = tangency_portfolio(mu_MV, sigma_MV, rf)
    stats_table["Risk-free rate"] = rf
    stats_table["Tangency Portfolio Weights"] = w_MV[TP_index, :].tolist()
    stats_table["Tickers"] = config["tickers"]

    portfolio_rets_insample = rets_insample.dot(w_MV[TP_index, :])

    capm_results = capm_decomposition(
        portfolio_rets_insample, rets_bench_insample[benchmark_ticker]
    )

    stats_table["In-Sample CAPM Beta"] = capm_results["Portfolio Beta"]
    stats_table["In-Sample CAPM Alpha"] = capm_results["Portfolio Alpha"]
    stats_table["In-Sample Expected Systematic Return"] = capm_results[
        "Expected Systematic Return"
    ]
    stats_table["In-Sample Expected Idiosyncratic Return"] = capm_results[
        "Expected Idiosyncratic Return"
    ]
    stats_table["In-Sample Expected Total Excess Return"] = capm_results[
        "Expected Total Excess Return"
    ]

    capm_analysis_result = capm_analysis(capm_results, config)
    stats_table["CAPM Analysis Result"] = capm_analysis_result

    if capm_analysis_result == 0:
        print("CAPM Analysis: Portfolio is healthy.")
    elif capm_analysis_result == 0.5:
        print("CAPM Analysis: Portfolio yellow signal.")
    elif capm_analysis_result == 1:
        print("CAPM Analysis: Portfolio red signal.")

    config["risk_aversion"] = config.get("risk_aversion", 1.0) + capm_analysis_result

    plot_mean_variance_frontier(
        sigma_MV, mu_MV, sigma, mu, TP_index=TP_index, rf=rf, plot_picture=PLOT_PICTURE
    )

    ## Out-of-sample analysis
    if tickers_data_out_sample is None or benchmark_data_out_sample is None:
        print("No out-of-sample data available.")
        return True

    rets_outsample = price_to_returns(
        tickers_data_out_sample["Adj Close"], log=config.get("log_returns", False)
    )
    rets_bench_outsample = price_to_returns(
        benchmark_data_out_sample["Adj Close"], log=config.get("log_returns", False)
    )

    portfolio_rets_outofsample = rets_outsample.dot(w_MV[TP_index, :])

    capm_results_outofsample = capm_decomposition(
        portfolio_rets_outofsample, rets_bench_outsample[benchmark_ticker]
    )

    stats_table["Out-of-Sample CAPM Beta"] = capm_results_outofsample["Portfolio Beta"]
    stats_table["Out-of-Sample CAPM Alpha"] = capm_results_outofsample[
        "Portfolio Alpha"
    ]
    stats_table["Out-of-Sample Expected Systematic Return"] = capm_results_outofsample[
        "Expected Systematic Return"
    ]
    stats_table["Out-of-Sample Expected Idiosyncratic Return"] = (
        capm_results_outofsample["Expected Idiosyncratic Return"]
    )
    stats_table["Out-of-Sample Expected Total Excess Return"] = (
        capm_results_outofsample["Expected Total Excess Return"]
    )

    # Compare expected vs realised performance
    expected_mu = mu_MV[TP_index, 0]
    expected_sigma = sigma_MV[TP_index, 0]

    # Run the portfolio
    portfolio_stats = {
        "expected_return": expected_mu,
        "expected_volatility": expected_sigma,
        "risk_free_rate": rf,
        "risk_aversion": config.get("risk_aversion", 1.0),
    }
    future_returns = rets_outsample.dot(w_MV[TP_index, :])
    port_return, backtest_stats = run_portfolio(future_returns, config, portfolio_stats, rf, plot_picture=PLOT_PICTURE)

    stats_table["Portfolio Expected Return (µ)"] = expected_mu
    stats_table["Portfolio Realised Return"] = backtest_stats["CAGR"]
    stats_table["Portfolio Expected Volatility (σ)"] = expected_sigma
    stats_table["Portfolio Realised Volatility"] = backtest_stats["Annualized Volatility"]
    stats_table["Portfolio Expected Sharpe Ratio"] = (expected_mu - rf) / expected_sigma
    stats_table["Portfolio Realised Sharpe Ratio"] = backtest_stats["Sharpe Ratio"]
    stats_table["Portfolio Win Rate"] = backtest_stats["Win Rate"]
    stats_table["Portfolio Average Gain"] = backtest_stats["Average Gain"]
    stats_table["Portfolio Average Loss"] = backtest_stats["Average Loss"]
    stats_table["Portfolio Max Drawdown"] = backtest_stats["Max Drawdown"]
    stats_table["Overall Portfolio Total Gain"] = (port_return[-1] - config["start_capital"]) / config["start_capital"]
    stats_table["Asset based Portfolio Total Return"] = backtest_stats["Total Return"]
    stats_table["Portfolio split"] = backtest_stats["Weight in risky portfolio (y)"]

    save_stats(stats_table, config["project_name"])


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def read_config(path: str) -> dict:
    """Read YAML config file and return as a dictionary."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    config = read_config("input_data/config_file.yaml")
    run_analysis(config)
