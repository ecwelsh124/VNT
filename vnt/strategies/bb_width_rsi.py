# vnt/strategies/bb_width_rsi.py

from itertools import product
import numpy as np
from typing import Iterator, NamedTuple, Optional, Tuple, Any
import logging

from core.strategy import Strategy
from helpers.helper_funcs import cached_indicator, rsi, bb
from core.enums import CandleEnum

logger = logging.getLogger(__name__)

class IndicatorSettings(NamedTuple):
    rsi_is_above: Optional[np.ndarray]
    rsi_is_below: Optional[np.ndarray]
    rsi_length: np.ndarray
    bb_length: np.ndarray
    bb_std_dev: np.ndarray
    bb_width_threshold: np.ndarray

class BBWidthRSI(Strategy):
    def __init__(
        self,
        backtest_settings_tuple,
        exchange_settings_tuple,
        static_os_tuple,
        direction, 
        ):
        super().__init__(
            backtest_settings_tuple,
            exchange_settings_tuple,
            static_os_tuple,
        )
        self.direction = direction

    def set_indicator_parameters(self, parameters: Tuple[Any, ...]):
        # Unpack indicator parameters based on direction
        if self.direction == 'long':
            self.cur_ind_set_tuple = IndicatorSettings(
                rsi_is_above=None,
                rsi_is_below=parameters[0],
                rsi_length=parameters[1],
                bb_length=parameters[2],
                bb_std_dev=parameters[3],
                bb_width_threshold=parameters[4],
            )
        elif self.direction == 'short':
            self.cur_ind_set_tuple = IndicatorSettings(
                rsi_is_above=parameters[0],
                rsi_is_below=None,
                rsi_length=parameters[1],
                bb_length=parameters[2],
                bb_std_dev=parameters[3],
                bb_width_threshold=parameters[4],
            )
        elif self.direction == 'both':
            self.cur_ind_set_tuple = IndicatorSettings(
                rsi_is_above=parameters[0],
                rsi_is_below=parameters[1],
                rsi_length=parameters[2],
                bb_length=parameters[3],
                bb_std_dev=parameters[4],
                bb_width_threshold=parameters[5],
            )
        else:
            raise ValueError(f"Invalid direction: {self.direction}")

    def set_entries_exits_array(self, candles):
        rsi_is_above = self.cur_ind_set_tuple.rsi_is_above
        rsi_is_below = self.cur_ind_set_tuple.rsi_is_below
        rsi_length = int(self.cur_ind_set_tuple.rsi_length)
        bb_length = int(self.cur_ind_set_tuple.bb_length)
        bb_std_dev = self.cur_ind_set_tuple.bb_std_dev
        bb_width_threshold = self.cur_ind_set_tuple.bb_width_threshold
        
        close_prices = candles.candle_close_prices
        open_prices = candles.candle_open_prices

        # Compute RSI
        self.rsi_values = cached_indicator(rsi, close_prices, rsi_length)
        self.rsi_values = np.around(self.rsi_values, 1)

        # Compute Bollinger Bands
        self.middle_bb, self.upper_bb, self.lower_bb = cached_indicator(bb, close_prices, bb_length, bb_std_dev)

        # Compute Bollinger Band width
        bb_width = (self.upper_bb - self.lower_bb) / self.middle_bb

        # Ensure there are no NaN values before comparison
        valid_indices = ~np.isnan(self.middle_bb) & ~np.isnan(self.upper_bb) & ~np.isnan(self.lower_bb) & ~np.isnan(self.rsi_values)
        
        # Conditions
        bb_width_greater_than_threshold = bb_width > bb_width_threshold

        # Previous close
        prev_close = np.roll(close_prices, 1)
        prev_close[0] = np.nan

        # Previous RSI
        prev_rsi = np.roll(self.rsi_values, 1)
        prev_rsi[0] = np.nan

        # Current candle direction
        current_candle_green = close_prices > open_prices
        current_candle_red = close_prices < open_prices

        # Long conditions
        if self.cur_ind_set_tuple.rsi_is_below is not None:
            long_entries = (
                bb_width_greater_than_threshold &
                (prev_close < self.lower_bb) &
                (prev_rsi < rsi_is_below) &
                current_candle_green &
                valid_indices
            )
        else:
            long_entries = np.full_like(close_prices, False)

        # Short conditions
        if self.cur_ind_set_tuple.rsi_is_above is not None:
            short_entries = (
                bb_width_greater_than_threshold &
                (prev_close > self.upper_bb) &
                (prev_rsi > rsi_is_above) &
                current_candle_red &
                valid_indices
            )
        else:
            short_entries = np.full_like(close_prices, False)

        # Set entries and exits based on direction
        if self.direction == 'long':
            self.entries = long_entries
            self.entries_short = np.full_like(self.entries, False)
        elif self.direction == 'short':
            self.entries = np.full_like(short_entries, False)
            self.entries_short = short_entries
        elif self.direction == 'both':
            self.entries = long_entries
            self.entries_short = short_entries
        else:
            logger.error(f'Invalid direction: {self.direction}')
            raise ValueError(f'Invalid direction: {self.direction}')
    

    # Optional: Implement a method to generate parameter combinations
    def generate_parameter_combinations(
        self,
        dos_tuple,
        ind_set_tuple,
    ) -> Iterator[Tuple[Any, ...]]:
        # Generate all combinations of dynamic order settings
        dos_params = [getattr(dos_tuple, field) for field in dos_tuple._fields]
        dos_combinations = list(product(*dos_params))

        # Generate indicator parameter combinations based on strategy direction
        ind_params_list = []
        
        # Add common indicator parameters
        ind_params_list.append(getattr(ind_set_tuple, 'rsi_length'))
        ind_params_list.append(getattr(ind_set_tuple, 'bb_length'))
        ind_params_list.append(getattr(ind_set_tuple, 'bb_std_dev'))
        ind_params_list.append(getattr(ind_set_tuple, 'bb_width_threshold'))
        
        # Add relevant RSI parameters based on strategy direction
        if self.direction == 'long':
            rsi_is_below = getattr(ind_set_tuple, 'rsi_is_below')
            ind_params_list.insert(0, rsi_is_below) # Insert at the beginning
        elif self.direction == 'short':
            rsi_is_above = getattr(ind_set_tuple, 'rsi_is_above')
            ind_params_list.insert(0, rsi_is_above)  # Insert at the beginning
        elif self.direction == 'both':
            rsi_is_above = getattr(ind_set_tuple, 'rsi_is_above')
            rsi_is_below = getattr(ind_set_tuple, 'rsi_is_below')
            ind_params_list.insert(0, rsi_is_below)
            ind_params_list.insert(0, rsi_is_above)
        else:
            raise ValueError(f"Invalid direction: {self.direction}")
        
        ind_combinations = list(product(*ind_params_list))

        total_combinations = len(dos_combinations) * len(ind_combinations)
        settings_indices = np.arange(total_combinations)

        idx = 0
        for dos_combination in dos_combinations:
            for ind_combination in ind_combinations:
                # Exclude combinations where RSI thresholds are invalid
                if ind_combination[0] > ind_combination[1]:
                    continue
                yield (settings_indices[idx],) + dos_combination + ind_combination
                idx += 1

    def run_backtest(self, candles):
        # Use the base class run_backtest or implement custom logic
        result = super().run_backtest(candles)
        
        return result