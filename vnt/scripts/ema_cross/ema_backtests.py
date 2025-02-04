# vnt/scripts/run_backtests.py

import argparse
from backtesters.bt_multi_bt import run_df_backtest
from strategies.ema_cross import EMACross, IndicatorSettings
from core.enums import (
    BacktestSettings,
    DynamicOrderSettings,
    ExchangeSettings,
    StaticOrderSettings,
    LeverageModeType,
    PositionModeType,
    IncreasePositionType,
    LeverageStrategyType,
    StopLossStrategyType,
    TakeProfitStrategyType,
    TrailingSLStrategyType,
)
import numpy as np
import pickle

def main():
    parser = argparse.ArgumentParser(description="Run backtests for trading strategies.")
    parser.add_argument('--data', type=str, required=True, help='Path to the candlestick data file.')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use for backtesting.')
    args = parser.parse_args()

    # Load candle data
    candle_data_file = '/Users/evan/Documents/Algo Trading/vnt/data/5m_2023.pkl'  # Update the path accordingly
    with open(candle_data_file, 'rb') as f:
        candles = pickle.load(f)

    # Define backtest settings (You can modify these or load from a config file)
    backtest_settings = BacktestSettings(
        gains_pct_filter=0.0,
        qf_filter=0.0,
        total_trade_filter=0,
    )

    # Define exchange settings
    exchange_settings = ExchangeSettings(
        asset_tick_step=0.0001,
        leverage_mode=LeverageModeType.Cross,
        leverage_tick_step=0.1,
        limit_fee_pct=0.0002,
        market_fee_pct=0.0004,
        max_asset_size=1000.0,
        max_leverage=100.0,
        min_asset_size=0.001,
        min_leverage=1.0,
        mmr_pct=0.005,
        position_mode=PositionModeType.OneWayMode,
        price_tick_step=0.0001,
    )

    # Define static order settings
    static_os_tuple = StaticOrderSettings(
        increase_position_type=IncreasePositionType.PctAccountEntrySize,
        leverage_strategy_type=LeverageStrategyType.Static,
        sl_strategy_type=StopLossStrategyType.SLBasedOnCandleBody,
        sl_to_be_bool=False,
        starting_bar=0,
        starting_equity=10000.0,
        static_leverage=1.0,
        tp_fee_type='market',
        tp_strategy_type=TakeProfitStrategyType.RiskReward,
        trailing_sl_strategy_type=TrailingSLStrategyType.Nothing,
    )

    # Define dynamic order settings
    dos_tuple = DynamicOrderSettings(
        account_pct_risk_per_trade=np.array([1.0]),
        max_trades=np.array([1]),
        risk_reward=np.array([2.0]),
        sl_based_on_add_pct=np.array([0.02]),
        sl_based_on_lookback=np.array([5]),
        sl_bcb_type=np.array([0]),
        sl_to_be_cb_type=np.array([0]),
        sl_to_be_when_pct=np.array([0.0]),
        trail_sl_bcb_type=np.array([0]),
        trail_sl_by_pct=np.array([0.0]),
        trail_sl_when_pct=np.array([0.0]),
        settings_index=np.array([0]),
    )

    # Define indicator settings
    ind_set_tuple = IndicatorSettings(
        first_ema_length=np.array([10, 300, 10]),
        second_ema_length=np.array([10, 300, 10]),
    )

    # Run the backtest
    results = run_df_backtest(
        candles,
        EMACross,
        dos_tuple,
        ind_set_tuple,
        backtest_settings,
        exchange_settings,
        static_os_tuple,
        threads=args.threads,
    )

    # Process the results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
