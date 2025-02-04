# vnt/backtesters/bt_multi_bt.py

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Any
from core.strategy import Strategy
from core.enums import FootprintCandlesTuple, DynamicOrderSettings
from helpers.helper_funcs import reset_indicator_cache
import logging

logger = logging.getLogger(__name__)

def run_df_backtest(
    candles: FootprintCandlesTuple,
    strategy_class: type,
    dos_tuple: DynamicOrderSettings,
    ind_set_tuple: Any,
    backtest_settings,
    exchange_settings,
    static_os_tuple,
    direction,
    threads: int = 8,
) -> List[Any]:
    """
    Runs the backtest using multiprocessing.

    Args:
        candles (FootprintCandlesTuple): The market data.
        strategy_class (type): The strategy class to be tested.
        dos_tuple (DynamicOrderSettings): Dynamic order settings.
        ind_set_tuple (Any): Indicator settings.
        backtest_settings: Backtest settings.
        exchange_settings: Exchange settings.
        static_os_tuple: Static order settings.
        threads (int): Number of threads to use.
        direction: Direction of the strategy.

    Returns:
        List[Any]: List of backtest results.
    """

    # Initialize the list to store results
    results = []

    # Initialize the strategy
    strategy = strategy_class(
        backtest_settings,
        exchange_settings,
        static_os_tuple,
        direction,
    )

    # Generate parameter combinations
    parameter_combinations = list(strategy.generate_parameter_combinations(dos_tuple, ind_set_tuple))

    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor(max_workers=threads) as executor:
        # Submit tasks to the executor
        future_to_params = {
            executor.submit(
                backtest_single_parameter_set,
                candles,
                strategy_class,
                params,
                backtest_settings,
                exchange_settings,
                static_os_tuple,
                direction,
            ): params for params in parameter_combinations
        }

        # Process completed futures as they finish
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")

    return results

def backtest_single_parameter_set(
    candles: FootprintCandlesTuple,
    strategy_class: type,
    parameters: Any,
    backtest_settings,
    exchange_settings,
    static_os_tuple,
    direction: str,
):
    """
    Runs the backtest for a single set of parameters.

    Args:
        candles (FootprintCandlesTuple): The market data.
        strategy_class (type): The strategy class to be tested.
        parameters (Any): Parameters for the strategy.
        backtest_settings: Backtest settings.
        exchange_settings: Exchange settings.
        static_os_tuple: Static order settings.

    Returns:
        Any: The result of the backtest.
    """
    try:
        # Reset the indicator cache to avoid memory buildup
        reset_indicator_cache()

        # Initialize the strategy
        strategy = strategy_class(
            backtest_settings,
            exchange_settings,
            static_os_tuple,
            direction,
        )

        # Set the parameters
        strategy.set_parameters(parameters)

        # Run the backtest
        result = strategy.run_backtest(candles)

        return result
    except Exception as e:
        logging.exception(f"Exception in backtest_single_parameter_set: {e}")
        return None
