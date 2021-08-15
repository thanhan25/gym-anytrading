import numpy as np

from .trading_env import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.trade_fee_bid_percent = 0.1 / 100  # unit
        self.trade_fee_ask_percent = 0.1 / 100  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0
        
        trade = False
        if action == Actions.Buy.value and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick] * (1 + np.random.uniform(low=-0.0005, high=0.0005))
            price_diff = current_price - last_trade_price
            step_reward += price_diff

        if action == Actions.Sell.value and self._position == Positions.Short:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick] * (1 + np.random.uniform(low=-0.0005, high=0.0005))
            price_diff = current_price - last_trade_price
            step_reward -= price_diff        

        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick] * (1 + np.random.uniform(low=-0.0005, high=0.0005))
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff
            if self._position == Positions.Short:
                step_reward -= price_diff
        return step_reward


    def _update_profit(self, action):

        if action == Actions.Buy.value and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick] * (1 + np.random.uniform(low=-0.0005, high=0.0005))    
            shares = round((self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price, 3)
            self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price             
       
        if action == Actions.Sell.value and self._position == Positions.Short:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick] * (1 + np.random.uniform(low=-0.0005, high=0.0005))    
            shares = round((self._total_profit * (1 - self.trade_fee_ask_percent)) / current_price, 3)
            self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * last_trade_price  
        
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick] * (1 + np.random.uniform(low=-0.0005, high=0.0005))

            if self._position == Positions.Long:
                shares = round((self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price, 3)
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
            if self._position == Positions.Short:
                shares = round((self._total_profit * (1 - self.trade_fee_bid_percent)) / current_price, 3)
                self._total_profit = (shares * (1 - self.trade_fee_ask_percent)) * last_trade_price


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1
            if position == Positions.Short:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[self.last_trade_tick] 
                shares = profit / current_price
                profit = shares * last_trade_price
            last_trade_tick = current_tick - 1

        return profit
