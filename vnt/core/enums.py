# vnt/core/enums.py

from enum import Enum
from typing import NamedTuple, Optional, Tuple
import numpy as np

class CandleBodyType(Enum):
    OpenDatetime = 0
    OpenTimestamp = 1
    CloseDatetime = 2
    CloseTimestamp = 3
    DurationSeconds = 4
    Open = 5
    High = 6
    Low = 7
    Close = 8
    UsdtVolume = 9
    AssetVolume = 10
    Nothing = 11

class IncreasePositionType(Enum):
    AmountEntrySize = 0
    PctAccountEntrySize = 1
    RiskAmountEntrySize = 2
    RiskPctAccountEntrySize = 3
    SmallestEntrySizeAsset = 4

class LeverageModeType(Enum):
    Cross = 0
    Isolated = 1

class LeverageStrategyType(Enum):
    Dynamic = 0
    Static = 1

class LoggerFuncType(Enum):
    Debug = 0
    Info = 1
    Warning = 2
    Error = 3

class OrderStatus(Enum):
    HitMaxTrades = 0
    EntryFilled = 1
    StopLossFilled = 2
    TakeProfitFilled = 3
    LiquidationFilled = 4
    MovedSLToBE = 5
    MovedTSL = 6
    MaxEquityRisk = 7
    RiskTooBig = 8
    CashUsedExceed = 9
    EntrySizeTooSmall = 10
    EntrySizeTooBig = 11
    PossibleLossTooBig = 12
    Nothing = 13

class PositionModeType(Enum):
    OneWayMode = 0
    BuySide = 1
    SellSide = 2
    HedgeMode = 3

class StringerFuncType(Enum):
    FloatToStr = 0
    LogDatetime = 1
    CandleBodyStr = 2
    OsToStr = 3

class StopLossStrategyType(Enum):
    Nothing = 0
    SLBasedOnCandleBody = 1

class TrailingSLStrategyType(Enum):
    Nothing = 0
    CBAboveBelow = 1
    PctAboveBelow = 2

class TakeProfitStrategyType(Enum):
    RiskReward = 0
    Provided = 1
    Nothing = 2

class TriggerDirectionType(Enum):
    Rise = 1
    Fall = 2

class FootprintCandlesTuple(NamedTuple):
    candle_open_datetimes: Optional[np.ndarray] = None
    candle_open_timestamps: Optional[np.ndarray] = None
    candle_close_datetimes: Optional[np.ndarray] = None
    candle_close_timestamps: Optional[np.ndarray] = None
    candle_durations_seconds: Optional[np.ndarray] = None
    candle_open_prices: Optional[np.ndarray] = None
    candle_high_prices: Optional[np.ndarray] = None
    candle_low_prices: Optional[np.ndarray] = None
    candle_close_prices: Optional[np.ndarray] = None
    candle_usdt_volumes: Optional[np.ndarray] = None
    candle_asset_volumes: Optional[np.ndarray] = None
    candle_trade_counts: Optional[np.ndarray] = None
    candle_deltas: Optional[np.ndarray] = None
    candle_delta_percents: Optional[np.ndarray] = None
    candle_buy_volumes: Optional[np.ndarray] = None
    candle_buy_counts: Optional[np.ndarray] = None
    candle_sell_volumes: Optional[np.ndarray] = None
    candle_sell_counts: Optional[np.ndarray] = None
    candle_cvds: Optional[np.ndarray] = None
    candle_pocs: Optional[np.ndarray] = None
    candle_high_lows: Optional[np.ndarray] = None
    prices_tuple: Optional[Tuple] = None
    prices_buy_vol_tuple: Optional[Tuple] = None
    prices_buy_count_tuple: Optional[Tuple] = None
    prices_sell_vol_tuple: Optional[Tuple] = None
    prices_sell_count_tuple: Optional[Tuple] = None
    prices_delta_tuple: Optional[Tuple] = None
    prices_delta_percent_tuple: Optional[Tuple] = None
    prices_volume_tuple: Optional[Tuple] = None
    prices_trade_count_tuple: Optional[Tuple] = None

class AccountState(NamedTuple):
    set_idx: int
    bar_index: int
    timestamp: int
    available_balance: float
    cash_borrowed: float
    cash_used: float
    equity: float
    fees_paid: float
    total_possible_loss: int
    realized_pnl: float
    total_trades: int

class BacktestSettings(NamedTuple):
    gains_pct_filter: float = -np.inf
    qf_filter: float = -np.inf
    total_trade_filter: int = -1
    
class IndicatorSettings(NamedTuple):
    empty: np.ndarray

class DynamicOrderSettings(NamedTuple):
    account_pct_risk_per_trade: np.ndarray
    max_trades: np.ndarray
    risk_reward: np.ndarray
    sl_based_on_add_pct: np.ndarray
    sl_based_on_lookback: np.ndarray
    sl_bcb_type: np.ndarray
    sl_to_be_cb_type: np.ndarray
    sl_to_be_when_pct: np.ndarray
    trail_sl_bcb_type: np.ndarray
    trail_sl_by_pct: np.ndarray
    trail_sl_when_pct: np.ndarray
    settings_index: np.ndarray = np.array([0])

class ExchangeSettings(NamedTuple):
    asset_tick_step: float
    leverage_mode: LeverageModeType
    leverage_tick_step: float
    limit_fee_pct: float
    market_fee_pct: float
    max_asset_size: float
    max_leverage: float
    min_asset_size: float
    min_leverage: float
    mmr_pct: float
    position_mode: PositionModeType
    price_tick_step: float

class StaticOrderSettings(NamedTuple):
    increase_position_type: IncreasePositionType
    leverage_strategy_type: LeverageStrategyType
    sl_strategy_type: StopLossStrategyType
    sl_to_be_bool: bool
    starting_bar: int
    starting_equity: float
    static_leverage: float
    tp_fee_type: str
    tp_strategy_type: TakeProfitStrategyType
    trailing_sl_strategy_type: TrailingSLStrategyType

class OrderResult(NamedTuple):
    average_entry: float = np.nan
    can_move_sl_to_be: bool = False
    entry_price: float = np.nan
    entry_size_asset: float = np.nan
    entry_size_usd: float = np.nan
    exit_price: float = np.nan
    leverage: float = np.nan
    liq_price: float = np.nan
    order_status: OrderStatus = OrderStatus.Nothing
    position_size_asset: float = np.nan
    position_size_usd: float = np.nan
    sl_pct: float = np.nan
    sl_price: float = np.nan
    tp_pct: float = np.nan
    tp_price: float = np.nan

class RejectedOrder(Exception):
    def __init__(self, msg: str = None):
        self.msg = msg

class DecreasePosition(Exception):
    def __init__(
        self,
        exit_fee_pct: Optional[float] = np.nan,
        exit_price: Optional[float] = np.nan,
        liq_price: Optional[float] = np.nan,
        msg: Optional[str] = None,
        order_status: Optional[OrderStatus] = None,
        sl_price: Optional[float] = np.nan,
    ):
        self.exit_fee_pct = exit_fee_pct
        self.exit_price = exit_price
        self.liq_price = liq_price
        self.msg = msg
        self.order_status = order_status
        self.sl_price = sl_price
