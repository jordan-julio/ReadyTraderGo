import asyncio
import itertools
import time
from typing import List
import numpy as np
import math
from queue import Queue
from threading import Timer

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, Side, MAXIMUM_ASK, MINIMUM_BID

POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
ACTIVE_ORDER_COUNT_LIMIT = 10
ACTIVE_VOLUME_LIMIT = 200

RATELIMIT = 24
MIN_VOLUME_OF_INTEREST = 20
SIG_STRETCH = 100.
MIN_BID_NEAREST_TICK = (
    MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
LOT_SIZE = 10


class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = dict()
        self.asks = dict()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.ftrmid = -1

        self.wmp = -1
        self.ask_prices = -1
        self.ask_volumes = -1
        self.bid_prices = -1
        self.bid_volumes = -1
        self.bestBid = -1
        self.bestAsk = -1

        self.awaitingCancel = set()
        self.count = 0
        self.tick = 0

    def on_order_book_update_message(self, instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes):
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        if (bid_volumes[0] == 0 and ask_volumes[0] == 0):
            return

        i = (bid_volumes[0] / (bid_volumes[0] + ask_volumes[0]))
        wmp = i*ask_prices[0] + (1-i)*bid_prices[0]

        if (instrument == 0):
            self.ftrmid = wmp
            return
        self.tick += 1
        if (self.tick % 4 == 0):
            self.count = 0

        bestAsk, bestBid = self.relevantPrices(
            ask_prices, ask_volumes, bid_prices, bid_volumes)
        self.wmp = wmp
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.bestBid = bestBid
        self.bestAsk = bestAsk

        self.deleteDeadTrades(bestBid, bestAsk)
        if (self.activeVolume() < 150):
            self.makeTrade(self.wmp, self.ask_prices, self.ask_volumes,
                           self.bid_prices, self.bid_volumes, self.bestBid, self.bestAsk)

    def makeTrade(self, midprice, ask_prices, ask_volumes, bid_prices, bid_volumes, bestBid, bestAsk):
        askV, bidV = self.manageVolumes(midprice)

        # if (askV <= 0 or bidV <= 0):
        # return
        askP, bidP = self.managePrices(midprice, bestBid, bestAsk)
        if (self.count < RATELIMIT-1):
            self.count += 2
            bidId, askId = next(self.order_ids), next(self.order_ids)
            self.logger.warning("1")
            if bidId != 0 and bidP not in (self.bid_price, 0):
                self.logger.warning("2")
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
            if askId != 0 and askP not in (self.ask_price, 0):
                self.logger.warning("3")
                self.send_cancel_order(self.ask_id)
                self.ask_id = 0
            self.logger.warning("4")
            self.send_insert_order(bidId, Side.BUY, bidP * TICK_SIZE_IN_CENTS,
                                   LOT_SIZE, Lifespan.GOOD_FOR_DAY)
            self.send_insert_order(
                askId, Side.SELL, askP * TICK_SIZE_IN_CENTS, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
            # self.send_insert_order(bidId, Side.BUY, bidP * TICK_SIZE_IN_CENTS,
            #                       bidV, Lifespan.GOOD_FOR_DAY)
            # self.send_insert_order(
            #    askId, Side.SELL, askP * TICK_SIZE_IN_CENTS, askV, Lifespan.GOOD_FOR_DAY)
            self.bids[bidId] = {"price": bidP, "volume": bidV}
            self.asks[askId] = {"price": askP, "volume": askV}

    def manageVolumes(self, wmp):
        if (len(self.bids) + len(self.asks) >= ACTIVE_ORDER_COUNT_LIMIT - 1):
            return (0, 0)
        if (self.activeVolume() >= ACTIVE_VOLUME_LIMIT):
            return (0, 0)

        left = ACTIVE_VOLUME_LIMIT - self.activeVolume()

        if (left < 20):
            return (0, 0)

        middiff = (self.ftrmid - wmp)/20.
        asks = self.sigmoid(self.position - middiff *
                            (POSITION_LIMIT))
        askVol = round(left*asks)
        bidVol = left-askVol
        if (self.position + bidVol >= POSITION_LIMIT):
            askVol += bidVol
            bidVol = 0
        if (self.position - askVol <= -POSITION_LIMIT):
            bidVol += askVol
            askVol = 0
        return (askVol, bidVol)

    def managePrices(self, midprice, bestBid, bestAsk):
        if (self.ftrmid >= bestBid and self.ftrmid <= bestAsk):
            askP = round(midprice + 0.6)
            bidP = round(midprice - 0.6)
        elif (self.ftrmid < bestBid):
            askP = round(midprice + 0.6)
            bidP = bestBid
        else:
            askP = bestAsk
            bidP = round(midprice - 0.6)
        self.ask_price = askP
        self.bid_price = bidP
        return (askP, bidP)

    def on_order_filled_message(self, client_order_id, price, volume) -> None:
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids.keys():
            self.position += volume
            self.bids[client_order_id]["volume"] -= volume
            self.send_hedge_order(next(self.order_ids),
                                  Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks.keys():
            self.position -= volume
            self.asks[client_order_id]["volume"] -= volume
            self.send_hedge_order(next(self.order_ids),
                                  Side.BID, MAX_ASK_NEAREST_TICK, volume)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int) -> None:
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if (client_order_id in self.bids):
                self.bid_id = 0
            elif (client_order_id in self.asks):
                self.ask_id = 0
            self.bids.pop(client_order_id, None)
            self.asks.pop(client_order_id, None)

    def activeVolume(self):
        sum = 0
        for k in self.bids.copy():
            sum += self.bids[k]["volume"]
        for k in self.asks.copy():
            sum += self.asks[k]["volume"]
        return sum

    def relevantPrices(self, ask_prices, ask_volumes, bid_prices, bid_volumes):
        i_list = np.where(np.asarray(ask_volumes) > MIN_VOLUME_OF_INTEREST)[0]
        j_list = np.where(np.asarray(bid_volumes) > MIN_VOLUME_OF_INTEREST)[0]
        if (i_list.size == 0 or j_list.size == 0):
            return (-1, -1)
        i = i_list[0]
        j = j_list[0]
        if (i >= len(ask_prices) or j >= len(bid_prices)):
            return (-1, -1)
        return (ask_prices[i], bid_prices[j])

    def deleteDeadTrades(self, bestBid, bestAsk):
        spread = bestAsk - bestBid
        for k, v in self.bids.copy().items():
            if (bestBid - v['price'] > 0 or (v['price'] - self.ftrmid) > 2 or spread > 3):
                if (self.count < RATELIMIT):
                    self.count += 1
                    self.awaitingCancel.add(k)
                    self.logger.warning(k)
                    self.send_cancel_order(k)
                else:
                    return
        for k, v in self.asks.copy().items():
            if (v['price'] - bestAsk > 0 or (self.ftrmid - v['price']) > 2 or spread > 3):
                if (self.count < RATELIMIT):
                    self.count += 1
                    self.awaitingCancel.add(k)
                    self.logger.warning(k)
                    self.send_cancel_order(k)
                else:
                    return

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x/SIG_STRETCH))

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

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
