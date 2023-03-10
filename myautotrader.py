import asyncio
import itertools
from typing import List
from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (
    MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class AutoTrader(BaseAutoTrader):

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.bid_volume = self.ask_volume = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        if instrument == Instrument.FUTURE:
            new_bid_price = bid_prices[0]
            new_ask_price = ask_prices[0]
            if self.position > 0:
                new_bid_price = max(
                    new_bid_price - TICK_SIZE_IN_CENTS, MIN_BID_NEAREST_TICK)
                new_ask_price = max(
                    new_ask_price - TICK_SIZE_IN_CENTS, MIN_BID_NEAREST_TICK)
            elif self.position < 0:
                new_bid_price = min(
                    new_bid_price + TICK_SIZE_IN_CENTS, MAX_ASK_NEAREST_TICK - TICK_SIZE_IN_CENTS)
                new_ask_price = min(
                    new_ask_price + TICK_SIZE_IN_CENTS, MAX_ASK_NEAREST_TICK)
            if self.bid_price != new_bid_price:
                if self.bid_id != 0:
                    self.send_cancel_order(self.bid_id)
                self.bid_id = next(self.order_ids)
                self.send_insert_order(
                    self.bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bid_price = new_bid_price
                self.bids.add(self.bid_id)
            if self.ask_price != new_ask_price:
                if self.ask_id != 0:
                    self.send_cancel_order(self.ask_id)
                self.ask_id = next(self.order_ids)
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price,
                                       LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)
                self.ask_price = new_ask_price

    def on_order_status_message(self, client_order_id: int, filled_volume: int, remaining_volume: int,
                                status_flags: int) -> None:
        if client_order_id in self.bids:
            self.bid_volume -= filled_volume
            if remaining_volume == 0:
                self.bids.discard(client_order_id)
        elif client_order_id in self.asks:
            self.ask_volume -= filled_volume
            if remaining_volume == 0:
                self.asks.discard(client_order_id)
        else:
            self.logger.warning(
                "received order status update for unknown order id %d", client_order_id)

        self.logger.info("order %d status update: filled volume=%d, remaining volume=%d, status flags=%d",
                         client_order_id, filled_volume, remaining_volume, status_flags)

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
            if self.position < POSITION_LIMIT:
                self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, min(
                    POSITION_LIMIT - self.position, volume))
        elif client_order_id in self.asks:
            self.position -= volume
            if self.position > -POSITION_LIMIT:
                self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, min(
                    POSITION_LIMIT + self.position, volume))

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        self.logger.info(
            "received trade ticks for instrument %d with sequence number %d", instrument, sequence_number)
        for i in range(len(bid_prices)):
            self.logger.info("bid price %d: %d, volume: %d",
                             i, bid_prices[i], bid_volumes[i])
        for i in range(len(ask_prices)):
            self.logger.info("ask price %d: %d, volume: %d",
                             i, ask_prices[i], ask_volumes[i])
