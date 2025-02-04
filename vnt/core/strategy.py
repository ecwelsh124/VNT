# vnt/core/strategy.py

from abc import ABC, abstractmethod
from typing import Any, Iterator, Tuple
import numpy as np
from logging import getLogger
from core.enums import (
    BacktestSettings,
    DynamicOrderSettings,
    ExchangeSettings,
    FootprintCandlesTuple,
    StaticOrderSettings,
    IndicatorSettings
)
from helpers.helper_funcs import get_qf_score
from memory_profiler import profile

logger = getLogger()

class Strategy(ABC):
    def __init__(
        self,
        backtest_settings_tuple: BacktestSettings,
        exchange_settings_tuple: ExchangeSettings,
        static_os_tuple: StaticOrderSettings,
        indicator_settings_NamedTuple: IndicatorSettings,
    ):
        self.backtest_settings_tuple = backtest_settings_tuple
        self.exchange_settings_tuple = exchange_settings_tuple
        self.static_os_tuple = static_os_tuple
        self.indicator_settings_NamedTuple = indicator_settings_NamedTuple
        self.entries: np.ndarray = np.array([])
        self.exits: np.ndarray = np.array([])
        self.long_short: str = 'long'  # or 'short'
        self.cur_dos_tuple: DynamicOrderSettings = None
        self.cur_ind_set_tuple: Any = None

    @abstractmethod
    def set_entries_exits_array(self, candles: FootprintCandlesTuple):
        """
        Calculate entries and exits based on strategy logic.
        To be implemented by each specific strategy.
        """
        pass

    @abstractmethod
    def set_indicator_parameters(self, parameters: Tuple[Any, ...]):
        """
        Set the indicator parameters for the strategy.
        To be implemented by each specific strategy.
        """
        pass

    def generate_parameter_combinations(
        self,
        dos_tuple: DynamicOrderSettings,
        ind_set_tuple: Any,
    ) -> Iterator[Tuple[Any, ...]]:
        dos_params = [getattr(dos_tuple, field) for field in dos_tuple._fields]
        ind_params = [getattr(ind_set_tuple, field) for field in ind_set_tuple._fields]

        # Generate all combinations
        dos_combinations = np.array(np.meshgrid(*dos_params)).T.reshape(-1, len(dos_params))
        ind_combinations = np.array(np.meshgrid(*ind_params)).T.reshape(-1, len(ind_params))
        
        # Generate settings_index
        total_combinations = len(dos_combinations) * len(ind_combinations)
        settings_indicies = np.arange(total_combinations)
        
        
        idx = 0
        for dos_combination in dos_combinations:
            for ind_combination in ind_combinations:
                yield (settings_indicies[idx],) + tuple(dos_combination) + tuple(ind_combination)
                idx += 1

    def set_dynamic_order_settings(self, parameters: Tuple[Any, ...]):
        """
        Set the dynamic order settings based on the parameters.

        Args:
            parameters (Tuple[Any, ...]): Parameters for dynamic order settings.
        """
        self.cur_dos_tuple = DynamicOrderSettings(*parameters)

    def set_current_indicator_settings(self, parameters: Tuple[Any, ...]):
        """
        Set the current indicator settings based on the parameters.

        Args:
            parameters (Tuple[Any, ...]): Parameters for indicator settings.
        """
        self.cur_ind_set_tuple = parameters

    def set_parameters(self, parameters: Tuple[Any, ...]):
        """
        Sets both dynamic order settings and indicator parameters.

        Args:
            parameters (Tuple[Any, ...]): Combined parameters.
        """
        settings_index = parameters[0]
        self.settings_index = settings_index
        
        dos_length = len(DynamicOrderSettings._fields)
        ind_length = len(self.indicator_settings_NamedTuple._fields)
        
        dos_params = parameters[1 : 1 + dos_length]
        ind_params = parameters[1 + dos_length : 1 + dos_length + ind_length]
        
        # Debug statements
        assert len(dos_params) == dos_length, f"Expected {dos_length} dynamic order settings, got {len(dos_params)}"
        assert len(ind_params) == ind_length, f'Expected {ind_length} indicator parameters, got {len(ind_params)}'
        
        self.set_dynamic_order_settings(dos_params)
        self.set_indicator_parameters(ind_params)
        
    # @profile use if kernal dies to check memory usage
    def run_backtest(self, candles: FootprintCandlesTuple):
        try:
            logger.info(f"Starting run_backtest for settings_index: {self.settings_index}")
             
            # Implement your backtesting logic here
            self.set_entries_exits_array(candles)

            close_prices = candles.candle_close_prices
            open_prices = candles.candle_open_prices
            timestamps = candles.candle_open_timestamps
            
            logger.debug(f"Number of candles: {len(close_prices)}")

            trades = []
            position = None
            entry_price = 0.0
            entry_time = 0
            equity_curve = [self.static_os_tuple.starting_equity]
            equity = self.static_os_tuple.starting_equity

            for i in range(len(close_prices) - 1):
                if i % 100000 == 0 and i > 0:
                    logger.info(f"Processed {i} candles.")
                
                
                if position is None:
                    if self.entries[i]:
                        # Open long position
                        entry_price = open_prices[i + 1]
                        entry_time = timestamps[i + 1]
                        position = 'long'
                        logger.debug(f"Opened long position at index {i+1}, price: {entry_price}")
                    elif self.entries_short[i]:
                        # Open short position
                        entry_price = open_prices[i + 1]
                        entry_time = timestamps[i + 1]
                        position = 'short'
                        logger.debug(f"Opened short position at index {i+1}, price: {entry_price}")
                elif position == 'long':
                    if self.exits[i]:
                        # Close long position
                        exit_price = close_prices[i]
                        exit_time = timestamps[i]
                        profit = exit_price - entry_price

                        # Record trade details
                        trades.append({
                            'position': 'long',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit,
                            'return_pct': (profit / entry_price) * 100,
                            'duration': exit_time - entry_time,
                        })
                        logger.debug(f"Closed long position at index {i}, price: {exit_price}, profit: {profit}")

                        equity += profit
                        equity_curve.append(equity)

                        # Reset position
                        position = None
                        
                elif position == 'short':
                    if self.exits_short[i]:
                        # Close short position
                        exit_price = close_prices[i]
                        exit_time = timestamps[i]
                        profit = entry_price - exit_price  # Profit calculation for short position

                        # Record trade details
                        trades.append({
                            'position': 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit,
                            'return_pct': (profit / entry_price) * 100,
                            'duration': exit_time - entry_time,
                        })
                        logger.debug(f"Closed short position at index {i}, price: {exit_price}, profit: {profit}")

                        equity += profit
                        equity_curve.append(equity)

                        # Reset position
                        position = None

            # Close any open positions at the end
            if position == 'long':
                exit_price = close_prices[-1]
                exit_time = timestamps[-1]
                profit = exit_price - entry_price

                # Record trade details
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'return_pct': (profit / entry_price) * 100,
                    'duration': exit_time - entry_time,
                })
                logger.debug(f"Closed long position at the end, price: {exit_price}, profit: {profit}")

                equity += profit
                equity_curve.append(equity)

                # Reset position
                position = None
                
            elif position == 'short':
                exit_price = close_prices[-1]
                exit_time = timestamps[-1]
                profit = entry_price - exit_price  # Profit calculation for short position

                # Record trade details
                trades.append({
                    'position': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'return_pct': (profit / entry_price) * 100,
                    'duration': exit_time - entry_time,
                })
                logger.debug(f"Closed short position at the end, price: {exit_price}, profit: {profit}")

                equity += profit
                equity_curve.append(equity)

                # Reset position
                position = None
                
            logger.info(f"Total trades executed: {len(trades)}")

            # Calculate performance metrics
            total_trades = len(trades)
            wins = [t for t in trades if t['profit'] > 0]
            losses = [t for t in trades if t['profit'] <= 0]
            total_profit = sum(t['profit'] for t in trades)
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            gains_pct = (equity - self.static_os_tuple.starting_equity) / self.static_os_tuple.starting_equity * 100

            logger.info(f"Total profit: {total_profit}, Win rate: {win_rate}%, Gains percentage: {gains_pct}%")
            
            # Calculate additional metrics
            average_winner_pct = np.mean([t['return_pct'] for t in wins]) if wins else 0
            average_loser_pct = np.mean([t['return_pct'] for t in losses]) if losses else 0
            average_return_per_trade = np.mean([t['return_pct'] for t in trades]) if trades else 0
            biggest_winner_pct = max([t['return_pct'] for t in trades]) if trades else 0
            biggest_loser_pct = min([t['return_pct'] for t in trades]) if trades else 0
            average_trade_duration = np.mean([t['duration'] for t in trades]) if trades else 0
            max_trade_duration = max([t['duration'] for t in trades]) if trades else 0

            # Calculate drawdowns
            equity_curve = np.array(equity_curve)
            peak_equity = np.maximum.accumulate(equity_curve)
            drawdowns = (peak_equity - equity_curve) / peak_equity * 100
            max_drawdown_pct = np.max(drawdowns)
            avg_drawdown_pct = np.mean(drawdowns)
            
            logger.info(f"Max drawdown: {max_drawdown_pct}%, Average drawdown: {avg_drawdown_pct}%")

            # Calculate qf_score (assuming it's based on equity curve)
            gains_pct_total = gains_pct
            pnl_array = np.array([t['profit'] for t in trades])
            qf_score = get_qf_score(gains_pct_total, pnl_array) if total_trades > 1 else 0.0

            logger.info(f"QF Score: {qf_score}")
            
            # Prepare the result
            result = {
                'settings_index': self.settings_index,
                'total_trades': total_trades,
                'total_profit': total_profit,
                'win_rate': win_rate,
                'gains_pct': gains_pct,
                'average_winner_pct': average_winner_pct,
                'average_loser_pct': average_loser_pct,
                'average_return_per_trade': average_return_per_trade,
                'biggest_winner_pct': biggest_winner_pct,
                'biggest_loser_pct': biggest_loser_pct,
                'average_trade_duration': average_trade_duration,
                'max_trade_duration': max_trade_duration,
                'max_drawdown_pct': max_drawdown_pct,
                'avg_drawdown_pct': avg_drawdown_pct,
                'qf_score': qf_score,
            }
            
            # # Dynamically include indicator settings in the result
            # if hasattr(self, 'cur_ind_set_tuple') and self.cur_ind_set_tuple is not None:
            #     indicator_settings_dict = self.cur_ind_set_tuple._asdict()
            #     result.update(indicator_settings_dict)
            #     logger.debug(f"Indicator settings: {indicator_settings_dict}")
            # if hasattr(self, 'cur_dos_tuple') and self.cur_dos_tuple is not None:
            #     dos_dict = self.cur_dos_tuple._asdict()
            #     result.update(dos_dict)
            #     logger.debug(f"Dynamic order settings: {dos_dict}")
                
            
            logger.info(f"Completed run_backtest for settings_index: {self.settings_index}")
            return result
        except Exception as e:
            logger.error(f"Error during run_backtest: {e}")
            return None