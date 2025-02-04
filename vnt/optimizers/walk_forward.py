# vnt/optimizers/walk_forward.py

import contextlib
import sys
import pandas as pd
import os
import re
import logging
from core.enums import FootprintCandlesTuple
from backtesters.bt_multi_bt import run_df_backtest
from tqdm import tqdm

logger = logging.getLogger(__name__)

def walk_forward_optimization(
    strategy_class,
    candles: FootprintCandlesTuple,
    train_timeframe: str,
    validation_timeframe: str,
    test_timeframe: str,
    backtest_settings,
    exchange_settings,
    static_os_tuple,
    dos_tuple,
    ind_set_tuple,
    direction: str,
    thread_count: int = 8,
    suppress_output: bool = True,
):
    """
    Performs Walk Forward Optimization (WFO) on the provided strategy and candle data.

    Args:
        strategy_class: The strategy class to be optimized.
        candles (FootprintCandlesTuple): The market data.
        train_timeframe (str): Timeframe for training period (e.g., '3M' for 3 months).
        validation_timeframe (str): Timeframe for validation period.
        test_timeframe (str): Timeframe for testing period.
        backtest_settings: Backtest settings.
        exchange_settings: Exchange settings.
        static_os_tuple: Static order settings.
        dos_tuple: Dynamic order settings.
        ind_set_tuple: Indicator settings.
        thread_count (int): Number of threads to use for backtesting.
        suppress_output (bool): Whether to suppress output during backtests.

    Returns:
        pd.DataFrame: DataFrame containing the results of the WFO.
    """

    logger.info("Starting walk_forward_optimization function")

    # Create a DataFrame to store the results
    master_df = pd.DataFrame()

    # Calculate the total number of seconds in a day
    seconds_in_a_day = 24 * 60 * 60

    # Calculate the number of intervals within a day
    intervals_in_a_day = seconds_in_a_day / candles.candle_durations_seconds[0]

    time_units = {
        'Y': 365 * intervals_in_a_day,
        'M': 30 * intervals_in_a_day,
        'W': 7 * intervals_in_a_day,
        'D': intervals_in_a_day,
        'H': intervals_in_a_day / 24,
    }

    # Convert timeframes to bars
    def calculate_bars(timeframe):
        num = int(re.findall(r'\d+', timeframe)[0])
        unit = re.findall(r'[A-Z]', timeframe)[0]
        return int(num * time_units[unit])
    
    key_columns = ['settings_index', 'iteration']

    def add_suffix_to_non_key_columns(df, suffix, key_columns):
        columns_to_suffix = [col for col in df.columns if col not in key_columns]
        rename_dict = {col: f"{col}{suffix}" for col in columns_to_suffix}
        df.rename(columns=rename_dict, inplace=True)
        return df

    warmup_bars = static_os_tuple.starting_bar
    train_bars = calculate_bars(train_timeframe)
    validation_bars = calculate_bars(validation_timeframe)
    test_bars = calculate_bars(test_timeframe)

    logger.info(f"Warmup bars: {warmup_bars}, Train bars: {train_bars}, Validation bars: {validation_bars}, Test bars: {test_bars}")

    # Convert datetime array to pandas datetime for easier manipulation
    datetime_array = pd.to_datetime(candles.candle_open_datetimes)

    # Calculate the total number of bars
    total_bars = len(datetime_array)
    logger.info(f"Total number of bars: {total_bars}")

    # Function for suppressing output
    @contextlib.contextmanager
    def suppress_stdout():
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    iteration = 0
    
    strategy = strategy_class(
        backtest_settings,
        exchange_settings,
        static_os_tuple,
        direction,
    )

    # Generate parameter combinations
    parameter_combinations = list(strategy.generate_parameter_combinations(dos_tuple, ind_set_tuple))
    
    print(f"Total parameter combinations: {len(parameter_combinations)}")
    print("Output suppressed:", suppress_output)

    # Iterate to slice the data
    for i in tqdm(range(train_bars + warmup_bars, total_bars - test_bars, test_bars)):

        # Slice the data for training, validation, and testing
        training_data = FootprintCandlesTuple(
            candle_open_datetimes=candles.candle_open_datetimes[i - train_bars - warmup_bars:i],
            candle_open_timestamps=candles.candle_open_timestamps[i - train_bars - warmup_bars:i],
            candle_durations_seconds=candles.candle_durations_seconds[i - train_bars - warmup_bars:i],
            candle_open_prices=candles.candle_open_prices[i - train_bars - warmup_bars:i],
            candle_high_prices=candles.candle_high_prices[i - train_bars - warmup_bars:i],
            candle_low_prices=candles.candle_low_prices[i - train_bars - warmup_bars:i],
            candle_close_prices=candles.candle_close_prices[i - train_bars - warmup_bars:i],
            candle_usdt_volumes=candles.candle_usdt_volumes[i - train_bars - warmup_bars:i],
            candle_asset_volumes=candles.candle_asset_volumes[i - train_bars - warmup_bars:i],
        )

        validation_data = FootprintCandlesTuple(
            candle_open_datetimes=candles.candle_open_datetimes[i - warmup_bars:i + validation_bars],
            candle_open_timestamps=candles.candle_open_timestamps[i - warmup_bars:i + validation_bars],
            candle_durations_seconds=candles.candle_durations_seconds[i - warmup_bars:i + validation_bars],
            candle_open_prices=candles.candle_open_prices[i - warmup_bars:i + validation_bars],
            candle_high_prices=candles.candle_high_prices[i - warmup_bars:i + validation_bars],
            candle_low_prices=candles.candle_low_prices[i - warmup_bars:i + validation_bars],
            candle_close_prices=candles.candle_close_prices[i - warmup_bars:i + validation_bars],
            candle_usdt_volumes=candles.candle_usdt_volumes[i - warmup_bars:i + validation_bars],
            candle_asset_volumes=candles.candle_asset_volumes[i - warmup_bars:i + validation_bars],
        )

        test_data = FootprintCandlesTuple(
            candle_open_datetimes=candles.candle_open_datetimes[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_open_timestamps=candles.candle_open_timestamps[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_durations_seconds=candles.candle_durations_seconds[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_open_prices=candles.candle_open_prices[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_high_prices=candles.candle_high_prices[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_low_prices=candles.candle_low_prices[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_close_prices=candles.candle_close_prices[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_usdt_volumes=candles.candle_usdt_volumes[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
            candle_asset_volumes=candles.candle_asset_volumes[i + validation_bars - warmup_bars:i + validation_bars + test_bars],
        )

        # Check for sufficient data length
        if len(validation_data.candle_open_datetimes) != (validation_bars + warmup_bars) or len(test_data.candle_open_datetimes) != (test_bars + warmup_bars):
            logger.warning(f"Skipping iteration {iteration} due to insufficient validation or test data")
            break

        # Run backtests for training, validation, and testing datasets
        if suppress_output:
            with suppress_stdout():
                logger.info(f"Running backtest for iteration {iteration}")
                training_results = run_df_backtest(
                    candles=training_data,
                    strategy_class=strategy_class,
                    dos_tuple=dos_tuple,
                    ind_set_tuple=ind_set_tuple,
                    backtest_settings=backtest_settings,
                    exchange_settings=exchange_settings,
                    static_os_tuple=static_os_tuple,
                    direction=direction,
                    threads=thread_count,
                )

                validation_results = run_df_backtest(
                    candles=validation_data,
                    strategy_class=strategy_class,
                    dos_tuple=dos_tuple,
                    ind_set_tuple=ind_set_tuple,
                    backtest_settings=backtest_settings,
                    exchange_settings=exchange_settings,
                    static_os_tuple=static_os_tuple,
                    direction=direction,
                    threads=thread_count,
                )

                testing_results = run_df_backtest(
                    candles=test_data,
                    strategy_class=strategy_class,
                    dos_tuple=dos_tuple,
                    ind_set_tuple=ind_set_tuple,
                    backtest_settings=backtest_settings,
                    exchange_settings=exchange_settings,
                    static_os_tuple=static_os_tuple,
                    direction=direction,
                    threads=thread_count,
                )
        else:
            training_results = run_df_backtest(
                candles=training_data,
                strategy_class=strategy_class,
                dos_tuple=dos_tuple,
                ind_set_tuple=ind_set_tuple,
                backtest_settings=backtest_settings,
                exchange_settings=exchange_settings,
                static_os_tuple=static_os_tuple,
                direction=direction,
                threads=thread_count,
            )

            validation_results = run_df_backtest(
                candles=validation_data,
                strategy_class=strategy_class,
                dos_tuple=dos_tuple,
                ind_set_tuple=ind_set_tuple,
                backtest_settings=backtest_settings,
                exchange_settings=exchange_settings,
                static_os_tuple=static_os_tuple,
                direction=direction,
                threads=thread_count,
            )

            testing_results = run_df_backtest(
                candles=test_data,
                strategy_class=strategy_class,
                dos_tuple=dos_tuple,
                ind_set_tuple=ind_set_tuple,
                backtest_settings=backtest_settings,
                exchange_settings=exchange_settings,
                static_os_tuple=static_os_tuple,
                direction=direction,
                threads=thread_count,
            )

        # Convert results to DataFrames
        train_df = pd.DataFrame(training_results)
        val_df = pd.DataFrame(validation_results)
        test_df = pd.DataFrame(testing_results)

        # Add iteration information
        train_df['iteration'] = iteration
        val_df['iteration'] = iteration
        test_df['iteration'] = iteration
        
        # Rename columns to indicate the dataset
        train_df = add_suffix_to_non_key_columns(train_df, '_train', key_columns)
        val_df = add_suffix_to_non_key_columns(val_df, '_val', key_columns)
        test_df = add_suffix_to_non_key_columns(test_df, '_test', key_columns)

        # Merge DataFrames on common columns
        combined_df = pd.merge(train_df, val_df, on=['settings_index', 'iteration'], how='inner')
        combined_df = pd.merge(combined_df, test_df, on=['settings_index', 'iteration'], how='inner')

        # Append to master DataFrame
        master_df = pd.concat([master_df, combined_df], ignore_index=True)

        iteration += 1

    # Optional: Drop unnecessary columns or perform further processing
    # For example, keep only relevant metrics
    columns_to_keep = [col for col in master_df.columns if any(metric in col for metric in [
        'settings_index', 'iteration', 'total_trades', 'gains_pct', 'qf_score', 'win_rate'
    ])]
    master_df = master_df[columns_to_keep]

    logger.info("Walk Forward Optimization completed")

    return master_df
