# strategies/ema_cross.py

import numpy as np
from core.strategy import Strategy
from core.enums import FootprintCandlesTuple, DynamicOrderSettings
from helpers.helper_funcs import cached_indicator, ema
from typing import Any, NamedTuple, Tuple, Iterator
from itertools import product

class IndicatorSettings(NamedTuple):
    first_ema_length: np.ndarray
    second_ema_length: np.ndarray

class EMACross(Strategy):
    def __init__(
        self,
        backtest_settings_tuple,
        exchange_settings_tuple,
        static_os_tuple,
        direction: str,
    ):
        super().__init__(
            backtest_settings_tuple,
            exchange_settings_tuple,
            static_os_tuple,
            indicator_settings_NamedTuple=IndicatorSettings,
        )
        self.direction = direction

    def set_indicator_parameters(self, parameters: Tuple[Any, ...]):
        self.cur_ind_set_tuple = IndicatorSettings(*parameters)

    def set_entries_exits_array(self, candles: FootprintCandlesTuple):
        first_ema_length = self.cur_ind_set_tuple.first_ema_length
        second_ema_length = self.cur_ind_set_tuple.second_ema_length

        close_prices = candles.candle_close_prices

        self.first_ema = cached_indicator(ema, close_prices, int(first_ema_length))
        self.second_ema = cached_indicator(ema, close_prices, int(second_ema_length))

        # Ensure there are no NaN values before comparison
        valid_indices = ~np.isnan(self.first_ema) & ~np.isnan(self.second_ema)

        # Define entry and exit signals based on the EMA crossover
        self.entries = np.zeros_like(close_prices, dtype=bool)
        self.exits = np.zeros_like(close_prices, dtype=bool)
        self.entries_short = np.zeros_like(close_prices, dtype=bool)
        self.exits_short = np.zeros_like(close_prices, dtype=bool)

        # Calculate entry and exit signals
        # Long entries: Fast EMA crosses above Slow EMA
        long_entries = np.zeros_like(close_prices, dtype=bool)
        long_entries[valid_indices] = (
            (self.first_ema[valid_indices] > self.second_ema[valid_indices]) &
            (np.roll(self.first_ema, 1)[valid_indices] <= np.roll(self.second_ema, 1)[valid_indices])
        )

        # Long exits: Fast EMA crosses below Slow EMA
        long_exits = np.zeros_like(close_prices, dtype=bool)
        long_exits[valid_indices] = (
            (self.first_ema[valid_indices] < self.second_ema[valid_indices]) &
            (np.roll(self.first_ema, 1)[valid_indices] >= np.roll(self.second_ema, 1)[valid_indices])
        )

        # Short entries: Fast EMA crosses below Slow EMA
        short_entries = np.zeros_like(close_prices, dtype=bool)
        short_entries[valid_indices] = (
            (self.first_ema[valid_indices] < self.second_ema[valid_indices]) &
            (np.roll(self.first_ema, 1)[valid_indices] >= np.roll(self.second_ema, 1)[valid_indices])
        )

        # Short exits: Fast EMA crosses above Slow EMA
        short_exits = np.zeros_like(close_prices, dtype=bool)
        short_exits[valid_indices] = (
            (self.first_ema[valid_indices] > self.second_ema[valid_indices]) &
            (np.roll(self.first_ema, 1)[valid_indices] <= np.roll(self.second_ema, 1)[valid_indices])
        )

        # Assign entries and exits based on direction
        if self.direction == 'long':
            self.entries = long_entries
            self.exits = long_exits
            self.entries_short = np.full_like(close_prices, False)
            self.exits_short = np.full_like(close_prices, False)
        elif self.direction == 'short':
            self.entries = np.full_like(close_prices, False)
            self.exits = np.full_like(close_prices, False)
            self.entries_short = short_entries
            self.exits_short = short_exits
        elif self.direction == 'both':
            self.entries = long_entries
            self.exits = long_exits
            self.entries_short = short_entries
            self.exits_short = short_exits
        else:
            raise ValueError(f"Invalid direction: {self.direction}")
    
    def generate_parameter_combinations(
        self,
        dos_tuple: DynamicOrderSettings,
        ind_set_tuple: IndicatorSettings,
    ) -> Iterator[Tuple[Any, ...]]:
        dos_params = [getattr(dos_tuple, field) for field in dos_tuple._fields]
        ind_params = [getattr(ind_set_tuple, field) for field in ind_set_tuple._fields]

        dos_combinations = list(product(*dos_params))
        ind_combinations = list(product(*ind_params))

        # Filter out invalid combinations
        filtered_ind_combinations = [
            combo for combo in ind_combinations if combo[0] != combo[1]
        ]

        total_combinations = len(dos_combinations) * len(filtered_ind_combinations)
        settings_indices = np.arange(total_combinations)

        idx = 0
        for dos_combination in dos_combinations:
            for ind_combination in filtered_ind_combinations:
                yield (settings_indices[idx],) + dos_combination + ind_combination
                idx += 1
    
    def run_backtest(self, candles: FootprintCandlesTuple):
        result = super().run_backtest(candles)
        
        return result
    
    

