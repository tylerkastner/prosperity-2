from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder, Trade, Listing, Observation

from typing import Dict, List, Any
import jsonpickle
import json
import string
import numpy as np
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()


class Trader:

    def trade_amethysts(self, amethyst_orders: OrderDepth, amethyst_pos):
      amethyst_results = []

      amethyst_buy_order_vol = 0
      amethyst_sell_order_vol = 0

      amethyst_buy_avail = 20 - amethyst_pos
      amethyst_sell_avail = -20 - amethyst_pos


      for ask, vol in sorted(amethyst_orders.sell_orders.items(), key=lambda x: x[0]):
        if ask < 10_000:
          vol = max(vol, -amethyst_buy_avail)

          # If we only take 1 profit, don't fill inventory
          if ask == 9_999 and -amethyst_buy_avail == vol:
            vol += 2

          if vol < 0:
            amethyst_results.append(Order("AMETHYSTS", ask, -vol))
            amethyst_buy_order_vol -= vol


        # Balance position if possible
        if ask == 10_000 and amethyst_pos + amethyst_buy_order_vol < 0:
          order_vol = max(vol, amethyst_pos + amethyst_buy_order_vol)
          amethyst_results.append(Order("AMETHYSTS", 10_000, -order_vol))
          amethyst_buy_order_vol -= order_vol


      for bid, vol in sorted(amethyst_orders.buy_orders.items(), key=lambda x:x[0], reverse=True):
        if bid > 10_000:
          vol = min(vol, -amethyst_sell_avail)

          # If we only take 1 profit, don't fill inventory
          if (bid == 10_001) and -amethyst_sell_avail == vol:
            vol -= 2

          if vol > 0:
            amethyst_results.append(Order("AMETHYSTS", bid, -vol))
            amethyst_sell_order_vol -= vol

        # Balance position if possible
        if bid == 10_000 and amethyst_pos + amethyst_sell_order_vol > 0:
          order_vol = min(amethyst_pos + amethyst_sell_order_vol, vol)
          amethyst_results.append(Order("AMETHYSTS", 10_000, -order_vol))
          amethyst_sell_order_vol -= order_vol


      # Market make using remaining volume

      good_amethyst_sell_orders = [o for o in amethyst_orders.sell_orders if o > 10_000]
      good_amethyst_buy_orders = [o for o in amethyst_orders.buy_orders if o < 10_000]

      lowest_good_amethyst_sell_order = min(good_amethyst_sell_orders, default=None)
      highest_good_amethyst_buy_order = max(good_amethyst_buy_orders, default=None)

      if highest_good_amethyst_buy_order:
        amethyst_buy_level = highest_good_amethyst_buy_order + 1 if highest_good_amethyst_buy_order < 9999 else 9999

        if amethyst_orders.buy_orders[highest_good_amethyst_buy_order] in [1]:
          amethyst_buy_level = highest_good_amethyst_buy_order
      else:
        amethyst_buy_level = 9995


      if lowest_good_amethyst_sell_order:
        amethyst_sell_level = lowest_good_amethyst_sell_order - 1 if lowest_good_amethyst_sell_order > 10001 else 10001

        if amethyst_orders.sell_orders[lowest_good_amethyst_sell_order] in [-1]:
          amethyst_sell_level = lowest_good_amethyst_sell_order

      else:
        amethyst_sell_level = 10_005


      amethyst_buy_volume = 20 - amethyst_pos - amethyst_buy_order_vol

      if amethyst_buy_volume > 0 and amethyst_buy_level > 9998:
        amethyst_buy_volume -= 2

      amethyst_sell_volume = -20 - amethyst_pos - amethyst_sell_order_vol

      if amethyst_sell_volume < 0 and amethyst_sell_level < 10_002:
        amethyst_sell_volume += 2


      if amethyst_sell_volume < 0:
        amethyst_results.append(Order("AMETHYSTS", amethyst_sell_level, amethyst_sell_volume))

      if amethyst_buy_volume > 0:
        amethyst_results.append(Order("AMETHYSTS", amethyst_buy_level, amethyst_buy_volume))

      return amethyst_results


    def trade_starfruit(self, starfruit_orders: OrderDepth, starfruit_pos, state, starfruit_micro_prices):
      starfruit_results = []

      starfruit_buy_avail = 20 - starfruit_pos
      starfruit_sell_avail = -20 - starfruit_pos

      starfruit_buy_order_vol = 0
      starfruit_sell_order_vol = 0

      buy_prices, buy_volumes = starfruit_orders.buy_orders.keys(), starfruit_orders.buy_orders.values()
      sell_prices, sell_volumes = starfruit_orders.sell_orders.keys(), starfruit_orders.sell_orders.values()

      buy_prices = np.array(list(buy_prices))
      buy_volumes = np.array(list(buy_volumes))
      sell_prices = np.array(list(sell_prices))
      sell_volumes = np.array(list(sell_volumes))

      starfruit_micro_price = (np.dot(buy_prices, buy_volumes) - np.dot(sell_prices, sell_volumes)) / (np.sum(buy_volumes) - np.sum(sell_volumes))
      starfruit_micro_prices.append(starfruit_micro_price.item())

      if len(starfruit_micro_prices) > 2:
        starfruit_micro_prices.pop(0)


      for ask, vol in sorted(starfruit_orders.sell_orders.items(), key=lambda x:x[0]):
        if ask < starfruit_micro_price:

          vol = max(vol, -starfruit_buy_avail)

          if (starfruit_micro_price - ask) < 1 and -starfruit_buy_avail == vol:
            vol += 3

          if vol < 0:
            starfruit_results.append(Order("STARFRUIT", ask, -vol))

            starfruit_buy_avail += vol
            starfruit_buy_order_vol -= vol


        if ask == starfruit_micro_price and starfruit_pos + starfruit_buy_order_vol < 0:

          order_vol = max(vol, starfruit_pos + starfruit_buy_order_vol)

          starfruit_results.append(Order("STARFRUIT", ask, -order_vol))

          starfruit_buy_order_vol -= vol


        if ask > starfruit_micro_price and (abs(starfruit_micro_price - ask) <= 1) and starfruit_pos + starfruit_buy_order_vol < -15:
          order_vol = max(starfruit_pos + starfruit_buy_order_vol, vol)
          starfruit_results.append(Order("STARFRUIT", ask, -order_vol))
          starfruit_buy_order_vol -= order_vol


      for bid, vol in sorted(starfruit_orders.buy_orders.items(), key=lambda x: x[0], reverse=True):
        if bid > starfruit_micro_price:
          vol = min(vol, -starfruit_sell_avail)

          if (bid - starfruit_micro_price) < 1 and -starfruit_sell_avail == vol:
            vol -= 3


          if vol > 0:
            starfruit_results.append(Order("STARFRUIT", bid, -vol))

            starfruit_sell_avail += vol
            starfruit_sell_order_vol -= vol


        if bid == starfruit_micro_price and starfruit_pos + starfruit_sell_order_vol > 0:
          order_vol = min(starfruit_pos + starfruit_sell_order_vol, vol)

          starfruit_results.append(Order("STARFRUIT", bid, -order_vol))

          starfruit_sell_order_vol -= order_vol

        if bid < starfruit_micro_price and (abs(starfruit_micro_price - bid) <= 1) and starfruit_pos + starfruit_sell_order_vol > 15:
          order_vol = min(starfruit_pos + starfruit_sell_order_vol, vol)
          starfruit_results.append(Order("STARFRUIT", bid, -order_vol))
          starfruit_sell_order_vol -= order_vol


      # Market make using remaining volume

      starfruit_buy_volume = 20 - starfruit_pos - starfruit_buy_order_vol
      starfruit_sell_volume = -20 - starfruit_pos - starfruit_sell_order_vol


      floor_micro_price = math.floor(starfruit_micro_price)
      ceil_micro_price = math.ceil(starfruit_micro_price)

      good_starfruit_sell_orders = [o for o in starfruit_orders.sell_orders if o > ceil_micro_price]
      good_starfruit_buy_orders = [o for o in starfruit_orders.buy_orders if o < floor_micro_price]


      sorted_good_sell_orders = sorted(good_starfruit_sell_orders)
      sorted_good_buy_orders = sorted(good_starfruit_buy_orders, reverse=True)


      if len(sorted_good_sell_orders) > 0:
        starfruit_sell_level = sorted_good_sell_orders[0] - 1 if sorted_good_sell_orders[0] >= starfruit_micro_price + 1 else sorted_good_sell_orders[0]

        if starfruit_orders.sell_orders[sorted_good_sell_orders[0]] in [-1, -2]:
          starfruit_sell_level = sorted_good_sell_orders[0]

      else:
        starfruit_sell_level = ceil_micro_price + 4

      if len(sorted_good_buy_orders) > 0:
        starfruit_buy_level = sorted_good_buy_orders[0] + 1 if sorted_good_buy_orders[0] <= starfruit_micro_price - 1 else sorted_good_buy_orders[0]

        if starfruit_orders.buy_orders[sorted_good_buy_orders[0]] in [1, 2]:
          starfruit_buy_level = sorted_good_buy_orders[0]
      else:
        starfruit_buy_level = floor_micro_price - 4


      if starfruit_buy_volume > 0 and (starfruit_micro_price - starfruit_buy_level) < 1:
        starfruit_buy_volume -= 3

      if starfruit_sell_volume < 0 and (starfruit_sell_level - starfruit_micro_price) < 1:
        starfruit_sell_volume += 3


      if starfruit_sell_volume < 0:
        starfruit_results.append(Order("STARFRUIT", starfruit_sell_level, starfruit_sell_volume))

      if starfruit_buy_volume > 0:
        starfruit_results.append(Order("STARFRUIT", starfruit_buy_level, starfruit_buy_volume))


      return starfruit_results


    def trade_orchids(self, orchids_orders: OrderDepth, orchids_pos, orchids_obs, state):
      orchids_results = []
      conversions = 0

      southBid = orchids_obs.bidPrice
      southAsk  = orchids_obs.askPrice
      transportFees = orchids_obs.transportFees
      exportTariff = orchids_obs.exportTariff
      importTariff = orchids_obs.importTariff

      fair_price_to_buy = southBid - transportFees - exportTariff
      fair_price_to_sell = transportFees + importTariff + southAsk

      if orchids_pos != 0:
        conversions = -orchids_pos

      if fair_price_to_sell <= math.ceil(southBid)-1:
        orchids_sell_level = math.ceil(southBid)-1
      else:
        orchids_sell_level = math.ceil(fair_price_to_sell)

      orchids_results.append(Order("ORCHIDS", orchids_sell_level, -100))

      return orchids_results, conversions


    def trade_basket(self, basket_orders: OrderDepth, chocolate_orders: OrderDepth, strawberry_orders: OrderDepth, roses_orders: OrderDepth, basket_pos, chocolate_pos, strawberry_pos, roses_pos):
      result = {'GIFT_BASKET': [], 'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': []}

      prodToOrders = {'GIFT_BASKET': basket_orders, 'CHOCOLATE': chocolate_orders, 'STRAWBERRIES': strawberry_orders, 'ROSES': roses_orders}
      prodToPos = {'GIFT_BASKET': basket_pos, 'CHOCOLATE': chocolate_pos, 'STRAWBERRIES': strawberry_pos, 'ROSES': roses_pos}


      prods = prodToOrders.keys()
      price_d = { prod: {} for prod in prods }

      for prod in prods:
        price_d[prod]['best_ask'] = min(prodToOrders[prod].sell_orders.keys())
        price_d[prod]['best_bid'] = max(prodToOrders[prod].buy_orders.keys())
        price_d[prod]['mid_price'] = (price_d[prod]['best_bid'] + price_d[prod]['best_ask']) / 2



      basket_price_estimate = 375 + 4*price_d['CHOCOLATE']['mid_price'] + 6*price_d['STRAWBERRIES']['mid_price'] + price_d['ROSES']['mid_price']


      def z_score_to_pos_limit(z):
        if z <= -.35:
          return True, 60

        if z >= .35:
          return True, -60


        return False, 0


      basket_buy_order_vol = 0
      basket_sell_order_vol = 0

      for ask, vol in sorted(basket_orders.sell_orders.items(), key=lambda x:x[0]):
        z_score = (ask - basket_price_estimate) / 78

        take_position, position_goal = z_score_to_pos_limit(z_score)

        buy_vol = position_goal - basket_pos - basket_buy_order_vol

        if take_position and buy_vol > 0:


          vol = max(vol, -buy_vol)

          if vol < 0:
            result["GIFT_BASKET"].append(Order("GIFT_BASKET", ask, -vol))

            basket_buy_order_vol -= vol

        elif z_score < 0 and (basket_pos+basket_buy_order_vol) < 0:
          vol = max(vol, basket_pos+basket_buy_order_vol)

          if vol < 0:
            result["GIFT_BASKET"].append(Order("GIFT_BASKET", ask, -vol))

            basket_buy_order_vol -= vol


      for bid, vol in sorted(basket_orders.buy_orders.items(), key=lambda x: x[0], reverse=True):
        z_score = (bid - basket_price_estimate) / 78


        take_position, position_goal = z_score_to_pos_limit(z_score)
        sell_vol = position_goal - basket_pos - basket_sell_order_vol

        if take_position and sell_vol < 0:


          vol = min(vol, -sell_vol)

          if vol > 0:
            result["GIFT_BASKET"].append(Order("GIFT_BASKET", bid, -vol))

            basket_sell_order_vol -= vol


        elif z_score > 0 and (basket_pos+basket_sell_order_vol) > 0:

          vol = min(vol, basket_pos+basket_sell_order_vol)

          if vol > 0:
            result["GIFT_BASKET"].append(Order("GIFT_BASKET", bid, -vol))

            basket_sell_order_vol -= vol



      lowest_good_sell_level = min(basket_orders.sell_orders.keys(), default=None)
      highest_good_buy_order = max(basket_orders.buy_orders.keys(), default=None)

      potential_sell_level = lowest_good_sell_level - 1
      potential_buy_level = highest_good_buy_order + 1



      # basket_buy_volume = 60 - basket_pos - basket_buy_order_vol
      # basket_sell_volume = -60 - basket_pos - basket_sell_order_vol



      sell_z_score = (potential_sell_level - basket_price_estimate) / 78
      take_position_sell, sell_position_goal = z_score_to_pos_limit(sell_z_score)
      sell_vol = sell_position_goal - basket_pos - basket_sell_order_vol


      # logger.print('potential_sell_level, sell_z_score, sell_position_goal')
      # logger.print(potential_sell_level, sell_z_score, sell_position_goal)


      if sell_vol < 0 and take_position_sell:
        result["GIFT_BASKET"].append(Order("GIFT_BASKET", potential_sell_level, sell_vol))


      elif sell_z_score > 0 and (basket_pos+basket_sell_order_vol) > 0:
        result["GIFT_BASKET"].append(Order("GIFT_BASKET", potential_sell_level, -(basket_pos+basket_sell_order_vol)))


      buy_z_score = (potential_buy_level - basket_price_estimate) / 78
      take_position_buy, buy_position_goal = z_score_to_pos_limit(buy_z_score)
      buy_vol = buy_position_goal - basket_pos - basket_buy_order_vol

      # logger.print('potential_buy_level, buy_z_score, buy_position_goal')
      # logger.print(potential_buy_level, buy_z_score, buy_position_goal)

      if take_position_buy and buy_vol > 0:
        result["GIFT_BASKET"].append(Order("GIFT_BASKET", potential_buy_level, buy_vol))

      elif buy_z_score < 0 and (basket_pos+basket_buy_order_vol) < 0:

        result["GIFT_BASKET"].append(Order("GIFT_BASKET", potential_buy_level, -(basket_pos+basket_buy_order_vol)))




      return result["GIFT_BASKET"], result["CHOCOLATE"], result["STRAWBERRIES"], result["ROSES"]


    def trade_roses(self, roses_orders, roses_pos, roses_short_until, roses_long_until, state):
      roses_result = []


      if roses_long_until is not None and state.timestamp < roses_long_until:
        if roses_pos < 60:

          best_roses_ask = min(roses_orders.sell_orders.keys())

          roses_result.append(Order("ROSES", best_roses_ask, 60-roses_pos))

      elif roses_short_until is not None and state.timestamp < roses_short_until:
        if roses_pos > -60:

          best_roses_bid = max(roses_orders.buy_orders.keys())

          roses_result.append(Order("ROSES", best_roses_bid, -60-roses_pos))

      elif roses_long_until is not None and roses_short_until is not None and (state.timestamp >= max(roses_short_until, roses_long_until)) and roses_pos != 0:
        if roses_pos > 0:

          best_roses_bid = max(roses_orders.buy_orders.keys())
          roses_result.append(Order("ROSES", best_roses_bid, -roses_pos))

        if roses_pos < 0:

          best_roses_ask = min(roses_orders.sell_orders.keys())
          roses_result.append(Order("ROSES", best_roses_ask, -roses_pos))

      return roses_result


    def trade_coupons(self, coconut_orders, coupon_orders, coconut_pos, coupon_pos, timestamp):
      result = { 'COCONUT': [], 'COCONUT_COUPON': [] }

      coconut_midprice = (min(coconut_orders.sell_orders.keys()) + max(coconut_orders.buy_orders.keys())) / 2

      # inverse normal CDF
      def phi(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


      dte = 246

      dte = dte - (timestamp / 1_000_000)



      T = dte / 252

      sigma = 0.16064

      d1 = (math.log(coconut_midprice / 10_000) + (T * sigma**2 / 2) ) / (sigma * math.sqrt(T))
      d2 = d1 - (sigma * math.sqrt(T))

      delta = phi(d1)

      premium = coconut_midprice * delta - 10_000 * phi(d2)


      coupon_order_vol = 0


      for ask, vol in coupon_orders.sell_orders.items():
        long_coupon_z_score = (ask - premium) / 13.5



        # buy coupon, short coconut
        if long_coupon_z_score < -.35:

          coupon_pos_goal = 600
          coupon_pos_delta = coupon_pos_goal - coupon_pos

          coupon_vol = max(vol, -coupon_pos_delta)

          result["COCONUT_COUPON"].append(Order("COCONUT_COUPON", ask, -coupon_vol))

          coupon_order_vol -= coupon_vol



      for bid, vol in coupon_orders.buy_orders.items():
        short_coupon_z_score = (premium - bid) / 13.5


        # short coupon
        if short_coupon_z_score < -.35:
          coupon_pos_goal = -600
          coupon_pos_delta = coupon_pos_goal - coupon_pos
          coupon_vol = min(vol, -coupon_pos_delta)

          result["COCONUT_COUPON"].append(Order("COCONUT_COUPON", bid, -coupon_vol))

          coupon_order_vol -= coupon_vol



      return result["COCONUT_COUPON"]


    def trade_choc(self, chocolate_orders, chocolate_pos, choc_long_until, state):
      choc_result = []


      if choc_long_until is not None and state.timestamp < choc_long_until:
        if chocolate_pos < 250:
          best_choc_ask = min(chocolate_orders.sell_orders.keys())
          choc_result.append(Order("CHOCOLATE", best_choc_ask, 250-chocolate_pos))


      elif choc_long_until is not None and (state.timestamp >= choc_long_until) and chocolate_pos != 0:
        if chocolate_pos > 0:

          best_choc_bid = max(chocolate_orders.buy_orders.keys())
          choc_result.append(Order("CHOCOLATE", best_choc_bid, -chocolate_pos))

        if chocolate_pos < 0:

          best_choc_ask = min(chocolate_orders.sell_orders.keys())
          choc_result.append(Order("CHOCOLATE", best_choc_ask, -chocolate_pos))

      return choc_result







    def run(self, state: TradingState):

        result = {}
        conversions = 0

        prev_trader_data = None

        if state.timestamp > 0:
          prev_trader_data = jsonpickle.loads(state.traderData)



        amethyst_orders = state.order_depths["AMETHYSTS"]

        amethyst_pos = 0
        if "AMETHYSTS" in state.position.keys():
          amethyst_pos = state.position["AMETHYSTS"]

        result["AMETHYSTS"] = self.trade_amethysts(amethyst_orders, amethyst_pos)


        starfruit_orders = state.order_depths["STARFRUIT"]
        starfruit_pos = 0
        if "STARFRUIT" in state.position.keys():
          starfruit_pos = state.position["STARFRUIT"]

        result["STARFRUIT"] = self.trade_starfruit(starfruit_orders, starfruit_pos, state, [])


        orchids_orders = state.order_depths["ORCHIDS"]
        orchids_pos = 0
        if "ORCHIDS" in state.position.keys():
          orchids_pos = state.position["ORCHIDS"]
        if "ORCHIDS" in state.observations.conversionObservations.keys():
          orchids_obs = state.observations.conversionObservations["ORCHIDS"]

        result["ORCHIDS"], conversions = self.trade_orchids(orchids_orders, orchids_pos, orchids_obs, state)



        basket_pos = state.position.get("GIFT_BASKET", 0)
        chocolate_pos = state.position.get("CHOCOLATE", 0)
        strawberry_pos = state.position.get("STRAWBERRIES", 0)
        roses_pos = state.position.get("ROSES", 0)

        basket_orders = state.order_depths.get("GIFT_BASKET")
        chocolate_orders = state.order_depths.get("CHOCOLATE")
        strawberry_orders = state.order_depths.get("STRAWBERRIES")
        roses_orders = state.order_depths.get("ROSES")

        result["GIFT_BASKET"], result["CHOCOLATE"], result["STRAWBERRIES"], result["ROSES"] = self.trade_basket(
          basket_orders, chocolate_orders, strawberry_orders, roses_orders, basket_pos, chocolate_pos, strawberry_pos, roses_pos)



        roses_market_trades = state.market_trades.get('ROSES', [])
        choc_market_trades = state.market_trades.get('CHOCOLATE', [])

        roses_long_until = None
        roses_short_until = None

        if prev_trader_data is not None and 'roses_long_until' in prev_trader_data:
          roses_long_until = prev_trader_data['roses_long_until']
        else:
          roses_long_until = None

        if prev_trader_data is not None and 'roses_short_until' in prev_trader_data:
          roses_short_until = prev_trader_data['roses_short_until']
        else:
          roses_short_until = None


        for trade in roses_market_trades:
          if trade.seller == 'Rhianna':
            roses_short_until = trade.timestamp + 20_000

          if trade.buyer == 'Rhianna':
            roses_long_until = trade.timestamp + 20_000

        result["ROSES"] = self.trade_roses(roses_orders, roses_pos, roses_short_until, roses_long_until, state)


        if prev_trader_data is not None and 'choc_long_until' in prev_trader_data:
          choc_long_until = prev_trader_data['choc_long_until']
        else:
          choc_long_until = None


        for trade in choc_market_trades:
          if trade.buyer == 'Vladimir':
            choc_long_until = trade.timestamp + 2_000

        result["CHOCOLATE"] = self.trade_choc(chocolate_orders, chocolate_pos, choc_long_until, state)




        coconut_pos = state.position.get("COCONUT", 0)
        coupon_pos = state.position.get("COCONUT_COUPON", 0)

        coconut_orders = state.order_depths.get("COCONUT")
        coupon_orders = state.order_depths.get("COCONUT_COUPON")


        result["COCONUT_COUPON"] = self.trade_coupons(coconut_orders, coupon_orders, coconut_pos, coupon_pos, state.timestamp)



        trader_data = { 'roses_long_until': roses_long_until, 'roses_short_until': roses_short_until, 'choc_long_until': choc_long_until }
        trader_data = jsonpickle.dumps(trader_data)





        logger.flush(state, result, conversions, trader_data)


        return result, conversions, trader_data



