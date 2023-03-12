# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import numpy as np
import pandas as pd
from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (
    MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.bidprices = []
        self.askprices = []
        self.last_order_prices = []
        self.activeorder = 0
        self.last_sold = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        if instrument == Instrument.FUTURE:
            # Determine the desired order book depth (in this example, we will use 5 levels)
            depth = 10
            # Calculate bid and ask price trends
            for i in sorted(bid_prices, reverse=True):
                if i != 0:
                    self.bidprices.append(i)
            for j in sorted(ask_prices, reverse=True):
                if j != 0:
                    self.askprices.append(j)
            if (len(self.bidprices) <= depth*depth or len(self.askprices) <= depth*depth):
                return
            if (len(self.bidprices) == 101):
                # self.logger.warning(self.bidprices)
                self.bidprices.pop(0)
                self.bidprices.pop(1)
                self.bidprices.pop(2)
                self.bidprices.pop(3)
                self.bidprices.pop(4)
            if (len(self.askprices) == 101):
                # self.logger.warning(self.askprices)
                self.askprices.pop(0)
                self.askprices.pop(1)
                self.askprices.pop(2)
                self.askprices.pop(3)
                self.askprices.pop(4)
            self.bidprices = sorted(self.bidprices, reverse=True)
            self.askprices = sorted(self.askprices)
            bid_prices_series = pd.Series(self.bidprices)
            ask_prices_series = pd.Series(self.askprices)
            # self.logger.warning(bid_prices_series)
            # self.logger.warning(ask_prices_series)
            # bid_trend = np.mean(
            #    self.bidprices[:depth]) - np.mean(self.bidprices[-depth:])
            # ask_trend = np.mean(
            #    self.askprices[-depth:]) - np.mean(self.askprices[:depth])
            bid_price_mean = np.mean(sorted(bid_prices, reverse=True)[:depth])
            ask_price_mean = np.mean(sorted(ask_prices)[:depth])
            bid_trend = (sum(self.bidprices[:depth]) /
                         depth - sum(self.bidprices[-depth:]) / depth)
            ask_trend = (sum(self.askprices[-depth:]) /
                         depth - sum(self.askprices[:depth]) / depth)

            # self.logger.warning(bid_trend)
            # self.logger.warning(ask_trend)
            price_adjustment = -1 * self.position * TICK_SIZE_IN_CENTS / LOT_SIZE
            # Determine the desired trade direction
            if bid_trend > 0 and ask_trend > 0:
                # Place a buy order
                order_direction = Side.BUY
                order_price = int((ask_prices[0] +
                                   price_adjustment if ask_prices[0] != 0 else 0))
            elif bid_trend > 0 and ask_trend < 0:
                # Place a buy order
                order_direction = Side.BUY
                order_price = int(bid_prices[-1] +
                                  price_adjustment if bid_prices[0] != 0 else 0)
            elif bid_trend < 0 and ask_trend > 0:
                # Place a sell order
                order_direction = Side.SELL
                order_price = int(ask_prices[0] +
                                  price_adjustment if ask_prices[0] != 0 else 0)
            else:
                # Do not place an order
                return

            # Calculate the order size based on available liquidity and risk management
            order_size = min(bid_volumes[0], ask_volumes[0])
            # Place the order
            order_id = next(self.order_ids)
            if order_direction == Side.BUY and self.activeorder == 0:
                self.bid_id = order_id
                # Check if the current market price is at least 10% higher than the buy price
                pricetodelete = 0
                for i in self.last_order_prices:
                    if ask_prices[0] >= i * 1.05 and self.activeorder == 0:
                        # filter the amount of things in prices
                        lenofsameprices = self.last_order_prices.count(i)
                        pricetodelete = i
                        self.logger.info(
                            "order id: %d, SELL, order price: %d, %d, GFD", order_id, ask_prices[0], LOT_SIZE * lenofsameprices)
                        self.send_insert_order(
                            order_id, Side.SELL, ask_prices[0], LOT_SIZE * lenofsameprices, Lifespan.GOOD_FOR_DAY)
                        self.activeorder = 1
                        self.asks.add(order_id)
                        self.last_order_prices = list(
                            [x for x in self.last_order_prices if x != pricetodelete])
                        # self.logger.info(self.last_order_prices)
                        self.last_sold = ask_prices[0]
                        return
                if (self.position + LOT_SIZE > POSITION_LIMIT):
                    order_size = (self.position + LOT_SIZE) - POSITION_LIMIT
                    if (order_size > 0):
                        return
                    self.logger.info(
                        "order id: %d, BUY, order price: %d, %d, GFD", order_id, order_price, LOT_SIZE)
                    self.activeorder = 1
                    self.send_insert_order(
                        order_id, order_direction, order_price * TICK_SIZE_IN_CENTS, order_size, Lifespan.GOOD_FOR_DAY)
                    self.last_order_prices.append(order_price)
                    return
                if (order_price >= self.last_sold and self.last_sold != 0):
                    return
                if (len(self.last_order_prices) == 0):
                    self.last_sold = 0
                self.logger.info(
                    "order id: %d, BUY, order price: %d, %d, GFD", order_id, order_price, LOT_SIZE)
                self.send_insert_order(
                    order_id, order_direction, order_price * TICK_SIZE_IN_CENTS, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.last_order_prices.append(order_price)
                self.bids.add(order_id)
                self.activeorder = 1
            elif order_direction == Side.SELL and self.position > -POSITION_LIMIT:
                self.ask_id = order_id
                self.logger.info(
                    "order id: %d, SELL, order price: %d, 10, GFD", order_id, order_price)
                self.send_insert_order(
                    order_id, order_direction, order_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.asks.add(order_id)
                self.activeorder = 1

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            # self.logger.warning(["volme", self.position, POSITION_LIMIT])
            self.bids.remove(client_order_id)
            self.send_hedge_order(next(self.order_ids),
                                  Side.ASK, MIN_BID_NEAREST_TICK, volume)
            self.activeorder = 0
        elif client_order_id in self.asks:
            self.position -= volume
            self.asks.remove(client_order_id)
            self.send_hedge_order(next(self.order_ids),
                                  Side.BID, MAX_ASK_NEAREST_TICK, volume)
            self.activeorder = 0

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
