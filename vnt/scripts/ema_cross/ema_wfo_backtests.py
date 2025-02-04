# vnt/main.py

from optimizers.walk_forward import walk_forward_optimization
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
    CandleBodyType,
)
from strategies.ema_cross import EMACross, IndicatorSettings
import numpy as np
import pickle

def main():
    # Load candle data from the pickle file
    candle_data_file = '/Users/evan/Documents/Algo Trading/vnt/data/5m_2023.pkl'  # Update the path accordingly
    with open(candle_data_file, 'rb') as f:
        candles = pickle.load(f)
        

    # Define backtest settings
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
        limit_fee_pct=0.0003,
        market_fee_pct=0.0006,
        max_asset_size=100.0,
        max_leverage=150.0,
        min_asset_size=0.001,
        min_leverage=1.0,
        mmr_pct=0.004,
        position_mode=PositionModeType.OneWayMode,
        price_tick_step=0.0001,
    )

    # Define static order settings
    static_os_tuple = StaticOrderSettings(
        increase_position_type=IncreasePositionType.RiskPctAccountEntrySize,
        leverage_strategy_type=LeverageStrategyType.Dynamic,
        sl_strategy_type=StopLossStrategyType.SLBasedOnCandleBody,
        sl_to_be_bool=False,
        starting_bar=100,
        starting_equity=1000.0,
        static_leverage=None,
        tp_fee_type='market',
        tp_strategy_type=TakeProfitStrategyType.RiskReward,
        trailing_sl_strategy_type=TrailingSLStrategyType.Nothing,
    )

    # Define dynamic order settings
    dos_tuple = DynamicOrderSettings(
        account_pct_risk_per_trade=np.array([2, 3, 4, 5]),
        max_trades=np.array([1, 2, 3]),
        risk_reward=np.array([3, 5, 10]),
        sl_based_on_add_pct=np.array([0.3, 0.5, 1.0]),
        sl_based_on_lookback=np.array([30]),
        sl_bcb_type=np.array([CandleBodyType.Low]),
        sl_to_be_cb_type=np.array([CandleBodyType.Nothing]),
        sl_to_be_when_pct=np.array([0]),
        trail_sl_bcb_type=np.array([CandleBodyType.Low]),
        trail_sl_by_pct=np.array([2, 4]),
        trail_sl_when_pct=np.array([2, 4]),
    )

    # Define indicator settings
    ind_set_tuple = IndicatorSettings(
        first_ema_length=np.arange(5, 51, 5),
        second_ema_length=np.arange(5, 51, 5),
    )

    # Define WFO hyperparameters
    train_timeframe = '3W'  # 3 Weeks
    validation_timeframe = '1W'  # 1 Weeks
    test_timeframe = '1W'  # 1 Weeks
    direction = 'long'
    
    # The number of threads to use
    thread_count = 8

    # Run Walk Forward Optimization
    wfo_results = walk_forward_optimization(
        strategy_class=EMACross,
        candles=candles,
        train_timeframe=train_timeframe,
        validation_timeframe=validation_timeframe,
        test_timeframe=test_timeframe,
        backtest_settings=backtest_settings,
        exchange_settings=exchange_settings,
        static_os_tuple=static_os_tuple,
        dos_tuple=dos_tuple,
        ind_set_tuple=ind_set_tuple,
        direction=direction,
        thread_count=thread_count,
        suppress_output=False,
    )

    # Process the results
    print(wfo_results)

    # Optionally, save the results to a file
    wfo_results.to_parquet('backtest_results/wfo_results_5m_2023.parquet')

if __name__ == '__main__':
    main()
