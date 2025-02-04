# vnt/order_handler/order.py

from typing import Optional
import numpy as np
from core.enums import ExchangeSettings

class OrderHandler:
    def __init__(
        self,
        exchange_settings: ExchangeSettings,
        long_short: str,
        market_fee_pct: float,
        tp_fee_pct: float,
    ):
        self.exchange_settings = exchange_settings
        self.long_short = long_short
        self.market_fee_pct = market_fee_pct
        self.tp_fee_pct = tp_fee_pct
        self.position_size_asset: float = 0.0
        self.average_entry: float = 0.0

    def calculate_fees(self, position_size_asset: float, entry_price: float, exit_price: Optional[float] = None) -> float:
        """
        Calculates the total fees for entering and exiting a position.

        Args:
            position_size_asset (float): The size of the position in assets.
            entry_price (float): The price at which the position was entered.
            exit_price (Optional[float]): The price at which the position is exited.

        Returns:
            float: The total fees paid.
        """
        fee_open = position_size_asset * entry_price * self.market_fee_pct
        fee_close = 0.0
        if exit_price is not None:
            fee_close = position_size_asset * exit_price * self.tp_fee_pct
        return fee_open + fee_close

    def calculate_pnl(self, entry_price: float, exit_price: float, position_size_asset: float) -> float:
        """
        Calculates the profit or loss for a trade.

        Args:
            entry_price (float): The entry price.
            exit_price (float): The exit price.
            position_size_asset (float): The size of the position in assets.

        Returns:
            float: The profit or loss.
        """
        if self.long_short == 'long':
            pnl = (exit_price - entry_price) * position_size_asset
        else:
            pnl = (entry_price - exit_price) * position_size_asset
        return pnl

    def enter_position(self, entry_price: float, position_size_asset: float):
        """
        Handles entering a new position.

        Args:
            entry_price (float): The entry price.
            position_size_asset (float): The size of the position in assets.
        """
        self.average_entry = entry_price
        self.position_size_asset += position_size_asset

    def exit_position(self, exit_price: float):
        """
        Handles exiting the current position.

        Args:
            exit_price (float): The exit price.
        """
        pnl = self.calculate_pnl(self.average_entry, exit_price, self.position_size_asset)
        fees = self.calculate_fees(self.position_size_asset, self.average_entry, exit_price)
        net_pnl = pnl - fees
        # Reset position
        self.position_size_asset = 0.0
        self.average_entry = 0.0
        return net_pnl
