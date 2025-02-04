# vnt/helpers/helper_funcs.py

import numpy as np
import pandas as pd
from typing import Callable, Any, Tuple, Dict

# Initialize the indicator cache
indicator_cache: Dict[Tuple[str, Tuple[Any, ...], Tuple[Tuple[str, Any], ...]], Any] = {}

def cached_indicator(indicator_function: Callable, *args, **kwargs) -> Any:
    """
    Caches the results of indicator calculations based on the function and its arguments.

    Args:
        indicator_function (Callable): The indicator function to be cached.
        *args: Positional arguments for the indicator function.
        **kwargs: Keyword arguments for the indicator function.

    Returns:
        Any: The result of the indicator calculation.
    """
    # Create a unique key based on the function name and its arguments
    hashable_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            # Convert the array to a bytes object
            arg_hash = hash(arg.tobytes())
            hashable_args.append(arg_hash)
        else:
            hashable_args.append(arg)
    cache_key = (indicator_function.__name__, tuple(hashable_args), frozenset(kwargs.items()))
    if cache_key in indicator_cache:
        return indicator_cache[cache_key]
    else:
        result = indicator_function(*args, **kwargs)
        indicator_cache[cache_key] = result
        return result
    
    
    
# Example indicator functions (replace with your actual indicator functions)
def ema(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculates the Exponential Moving Average (EMA) of a source array.

    Args:
        source (np.ndarray): The source data array.
        length (int): The period length for the EMA.

    Returns:
        np.ndarray: The EMA values.
    """
    ema_values = np.empty_like(source)
    ema_values[:] = np.nan  # Initialize with NaNs for periods before the EMA can be computed

    if length > len(source):
        return ema_values

    alpha = 2 / (length + 1)
    ema_values[length - 1] = np.mean(source[:length])  # Start EMA with SMA

    for i in range(length, len(source)):
        ema_values[i] = alpha * source[i] + (1 - alpha) * ema_values[i - 1]

    return ema_values

def sma(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculates the Simple Moving Average (SMA) of a source array.

    Args:
        source (np.ndarray): The source data array.
        length (int): The period length for the SMA.

    Returns:
        np.ndarray: The SMA values.
    """
    sma_values = np.convolve(source, np.ones(length), 'valid') / length
    sma_values = np.concatenate((np.full(length - 1, np.nan), sma_values))
    return sma_values

def rsi(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculates the Relative Strength Index (RSI) of a source array.

    Args:
        source (np.ndarray): The source data array (typically closing prices).
        length (int): The period length for the RSI.

    Returns:
        np.ndarray: The RSI values.
    """
    delta = np.diff(source)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)

    rsi_values = np.empty(len(source))
    rsi_values[:] = np.nan

    avg_gain = np.mean(up[:length])
    avg_loss = np.mean(down[:length])

    if avg_loss == 0:
        rsi_values[length] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_values[length] = 100 - (100 / (1 + rs))

    for i in range(length + 1, len(source)):
        gain = up[i - 1]
        loss = down[i - 1]

        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

        if avg_loss == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))

    return rsi_values

def bb(source: np.ndarray, length: int, mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Parameters:
    - source: np.ndarray
        The array of prices (e.g., closing prices).
    - length: int
        The period over which to calculate the bands.
    - mult: float
        The standard deviation multiplier (usually 2).

    Returns:
    - middle_bb: np.ndarray
        The middle Bollinger Band (Simple Moving Average).
    - upper_bb: np.ndarray
        The upper Bollinger Band.
    - lower_bb: np.ndarray
        The lower Bollinger Band.
    """
    # Ensure source is a NumPy array
    source = np.asarray(source, dtype=np.float64)

    # Calculate the Simple Moving Average (Middle Bollinger Band)
    middle_bb = np.full_like(source, fill_value=np.nan)
    cumulative_sum = np.cumsum(source, dtype=np.float64)
    cumulative_sum[length:] = cumulative_sum[length:] - cumulative_sum[:-length]
    middle_bb[length - 1:] = cumulative_sum[length - 1:] / length

    # Calculate the rolling standard deviation
    rolling_std = np.full_like(source, fill_value=np.nan)
    for i in range(length - 1, len(source)):
        window = source[i - length + 1:i + 1]
        rolling_std[i] = np.std(window, ddof=0)  # Population standard deviation

    # Calculate the upper and lower Bollinger Bands
    upper_bb = middle_bb + (rolling_std * mult)
    lower_bb = middle_bb - (rolling_std * mult)

    return middle_bb, upper_bb, lower_bb

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """
    Calculates the Average True Range (ATR) of price data.

    Args:
        high (np.ndarray): The high prices.
        low (np.ndarray): The low prices.
        close (np.ndarray): The closing prices.
        length (int): The period length for the ATR.

    Returns:
        np.ndarray: The ATR values.
    """
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])
    atr_values = np.empty(len(close))
    atr_values[:] = np.nan

    atr_values[length] = np.mean(tr[:length])

    for i in range(length + 1, len(close)):
        atr_values[i] = (atr_values[i - 1] * (length - 1) + tr[i - 1]) / length

    return atr_values

def get_qf_score(gains_pct: float, pnl_array: np.ndarray) -> float:
    """
    Calculates the Quant Freedom (QF) score based on gains percentage and P&L array.

    Args:
        gains_pct (float): The total gains percentage.
        pnl_array (np.ndarray): Array of P&L values for each trade.

    Returns:
        float: The QF score.
    """
    x = np.arange(1, len(pnl_array) + 1)
    y = pnl_array.cumsum()

    if len(x) < 2:
        return 0.0

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x

    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)

    if ss_tot == 0:
        return 0.0

    qf_score = 1 - (ss_res / ss_tot)

    if gains_pct <= 0:
        qf_score = -abs(qf_score)
    else:
        qf_score = abs(qf_score)

    return round(qf_score, 3)

def load_candles_from_parquet(file_path: str) -> Tuple[np.ndarray, ...]:
    """
    Loads candlestick data from a Parquet file.

    Args:
        file_path (str): The path to the Parquet file.

    Returns:
        Tuple[np.ndarray, ...]: A tuple containing arrays of open, high, low, close, volume, etc.
    """
    df = pd.read_parquet(file_path)
    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    volume = df['Volume'].values

    return open_prices, high_prices, low_prices, close_prices, volume

def float_to_str(num: float) -> str:
    """
    Converts a float to a string without scientific notation.

    Args:
        num (float): The float number.

    Returns:
        str: The string representation.
    """
    return format(num, 'f')

def log_datetime(timestamp: int) -> str:
    """
    Converts a timestamp to a human-readable date and time string.

    Args:
        timestamp (int): The timestamp in seconds.

    Returns:
        str: The formatted date and time string.
    """
    return pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')

def reset_indicator_cache():
    """
    Resets the indicator cache by clearing all cached values.
    """
    indicator_cache.clear()
    
