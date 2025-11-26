from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import talib
import numpy as np

class AlgoEvent:
    def __init__(self):
        self.lasttradetime = datetime(2000,1,1)
        
        # Data storage
        self.high_prices = []
        self.low_prices = []
        self.close_prices = []
        self.open_prices = []
        self.volumes = []
        
        # Optimized parameter settings
        self.fast_ema_period = 12       # Use EMA for better sensitivity
        self.slow_ema_period = 26
        self.signal_period = 9
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_ma_period = 20      # Volume filter
        
        # Risk management parameters
        self.base_position_size = 0.01
        self.max_position_size = 0.03
        self.risk_per_trade = 0.02      # 2% risk per trade
        self.max_daily_loss = 0.05      # 5% maximum daily loss
        self.max_drawdown = 0.15        # 15% maximum drawdown
        
        # Trading parameters
        self.atr_multiplier = 2.0
        self.trailing_stop_atr = 1.5    # Trailing stop
        self.min_interval = timedelta(minutes=30)
        self.max_holding_period = 7200  # 2 hours
        
        # Filter parameters
        self.min_adx = 25               # Trend strength filter
        self.min_volume_ratio = 1.2     # Volume amplification ratio
        
        # State variables
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        self.last_signal_time = None
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.trailing_stop = 0
        
        # Performance tracking
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        self.daily_pnl = 0
        self.last_reset_time = None
        self.peak_equity = 100000       # Assume initial capital
        self.current_equity = 100000
        
        # Signal confirmation
        self.pending_signal = None
        self.signal_confirmation_count = 0
        self.required_confirmations = 2

    def start(self, mEvt):
        self.myinstrument = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()
        self.evt.consoleLog("🚀 Optimized Multi-Factor Momentum Strategy Started - Enhanced Risk Control")

    def on_bulkdatafeed(self, isSync, bd, ab):
        if self.myinstrument not in bd:
            return
            
        current_time = bd[self.myinstrument]['timestamp']
        current_price = bd[self.myinstrument]['lastPrice']
        current_high = bd[self.myinstrument]['highPrice']
        current_low = bd[self.myinstrument]['lowPrice']
        current_open = bd[self.myinstrument]['openPrice']
        current_volume = bd[self.myinstrument]['volume']
        
        # Reset daily statistics
        self.reset_daily_stats(current_time)
        
        # Check risk management limits
        if not self.check_risk_limits():
            if self.current_position != 0:
                self.close_all_positions(current_price, current_time, "Risk Limit")
            return
        
        # Collect data
        self.collect_market_data(current_high, current_low, current_price, current_open, current_volume)
        
        # Check if sufficient data is available
        if len(self.close_prices) < 100:
            return
        
        # If holding position, check stop loss and take profit first
        if self.current_position != 0:
            self.check_exit_conditions(current_price, current_high, current_low, current_time)
        
        # Generate trading signals
        self.generate_trading_signals(current_price, current_time)

    def collect_market_data(self, high, low, close, open_price, volume):
        """Collect market data"""
        self.high_prices.append(high)
        self.low_prices.append(low)
        self.close_prices.append(close)
        self.open_prices.append(open_price)
        self.volumes.append(volume)
        
        # Maintain data length
        max_len = 150
        if len(self.close_prices) > max_len:
            self.high_prices = self.high_prices[-max_len:]
            self.low_prices = self.low_prices[-max_len:]
            self.close_prices = self.close_prices[-max_len:]
            self.open_prices = self.open_prices[-max_len:]
            self.volumes = self.volumes[-max_len:]

    def generate_trading_signals(self, current_price, current_time):
        """Generate trading signals - Enhanced version"""
        # Basic filters
        if not self.basic_filters(current_time):
            return
        
        # Calculate technical indicators
        indicators = self.calculate_indicators()
        if not indicators:
            return
            
        # Multi-factor filtering
        if not self.multi_factor_filter(indicators, current_price):
            self.pending_signal = None
            self.signal_confirmation_count = 0
            return
        
        # Signal confirmation mechanism
        current_signal = self.get_signal_direction(indicators, current_price)
        
        if current_signal != 0:
            if self.pending_signal == current_signal:
                self.signal_confirmation_count += 1
            else:
                self.pending_signal = current_signal
                self.signal_confirmation_count = 1
        else:
            self.pending_signal = None
            self.signal_confirmation_count = 0
        
        # Execute trade
        if (self.signal_confirmation_count >= self.required_confirmations and 
            self.pending_signal is not None and
            self.current_position == 0):
            
            if self.pending_signal == 1:
                self.open_long_position(current_price, indicators['atr'], current_time)
            elif self.pending_signal == -1:
                self.open_short_position(current_price, indicators['atr'], current_time)
            
            # Reset signal
            self.pending_signal = None
            self.signal_confirmation_count = 0

    def calculate_indicators(self):
        """Calculate all technical indicators"""
        try:
            close_array = np.array(self.close_prices)
            high_array = np.array(self.high_prices)
            low_array = np.array(self.low_prices)
            volume_array = np.array(self.volumes)
            
            # Trend indicators
            fast_ema = talib.EMA(close_array, timeperiod=self.fast_ema_period)
            slow_ema = talib.EMA(close_array, timeperiod=self.slow_ema_period)
            macd, macd_signal, macd_hist = talib.MACD(close_array, 
                                                     fastperiod=self.fast_ema_period,
                                                     slowperiod=self.slow_ema_period,
                                                     signalperiod=self.signal_period)
            
            # Momentum indicators
            rsi = talib.RSI(close_array, timeperiod=self.rsi_period)
            stoch_k, stoch_d = talib.STOCH(high_array, low_array, close_array,
                                          fastk_period=14, slowk_period=3, slowd_period=3)
            
            # Volatility indicators
            atr = talib.ATR(high_array, low_array, close_array, timeperiod=self.atr_period)
            adx = talib.ADX(high_array, low_array, close_array, timeperiod=14)
            
            # Volume indicators
            volume_ma = talib.SMA(volume_array, timeperiod=self.volume_ma_period)
            
            # Get latest values
            current_fast_ema = fast_ema[-1] if not np.isnan(fast_ema[-1]) else 0
            current_slow_ema = slow_ema[-1] if not np.isnan(slow_ema[-1]) else 0
            current_macd = macd[-1] if not np.isnan(macd[-1]) else 0
            current_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            current_stoch_k = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
            current_stoch_d = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
            current_volume = volume_array[-1] if len(volume_array) > 0 else 0
            current_volume_ma = volume_ma[-1] if not np.isnan(volume_ma[-1]) else 0
            
            return {
                'fast_ema': current_fast_ema,
                'slow_ema': current_slow_ema,
                'macd': current_macd,
                'macd_signal': current_macd_signal,
                'rsi': current_rsi,
                'stoch_k': current_stoch_k,
                'stoch_d': current_stoch_d,
                'atr': current_atr,
                'adx': current_adx,
                'volume': current_volume,
                'volume_ma': current_volume_ma,
                'volume_ratio': current_volume / current_volume_ma if current_volume_ma > 0 else 1
            }
        except Exception as e:
            self.evt.consoleLog(f"Indicator calculation error: {str(e)}")
            return None

    def multi_factor_filter(self, indicators, current_price):
        """Multi-factor filtering"""
        # Trend strength filter
        if indicators['adx'] < self.min_adx:
            return False
            
        # Volume filter
        if indicators['volume_ratio'] < self.min_volume_ratio:
            return False
            
        # Price position filter - avoid trading at extreme positions
        if indicators['rsi'] > 80 or indicators['rsi'] < 20:
            return False
            
        # Stochastic filter
        if indicators['stoch_k'] > 90 or indicators['stoch_d'] > 90:
            return False
        if indicators['stoch_k'] < 10 or indicators['stoch_d'] < 10:
            return False
            
        return True

    def get_signal_direction(self, indicators, current_price):
        """Determine signal direction"""
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA golden cross/death cross
        if indicators['fast_ema'] > indicators['slow_ema']:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # MACD signal
        if indicators['macd'] > indicators['macd_signal']:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # RSI signal
        if 40 <= indicators['rsi'] <= 70:
            bullish_signals += 0.5
        elif 30 <= indicators['rsi'] <= 60:
            bearish_signals += 0.5
            
        # Stochastic indicator
        if indicators['stoch_k'] > indicators['stoch_d']:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Comprehensive judgment
        if bullish_signals >= 3 and bearish_signals <= 1:
            return 1
        elif bearish_signals >= 3 and bullish_signals <= 1:
            return -1
        else:
            return 0

    def basic_filters(self, current_time):
        """Basic trading filters"""
        # Trading time interval
        if (self.last_signal_time and 
            (current_time - self.last_signal_time) < self.min_interval):
            return False
            
        # Position check
        if self.current_position != 0:
            return False
            
        return True

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk"""
        risk_amount = self.current_equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return self.base_position_size
            
        position_size = risk_amount / price_risk
        position_size = min(position_size, self.max_position_size)
        position_size = max(position_size, self.base_position_size)
        
        return round(position_size, 4)

    def open_long_position(self, price, atr, current_time):
        """Open long position - Enhanced version"""
        try:
            # Calculate stop loss and position size
            stop_loss_distance = atr * self.atr_multiplier
            stop_loss_price = max(price - stop_loss_distance, 0.01)
            position_size = self.calculate_position_size(price, stop_loss_price)
            
            order = AlgoAPIUtil.OrderObject()
            order.instrument = self.myinstrument
            order.openclose = 'open'
            order.buysell = 1
            order.ordertype = 0
            order.volume = position_size
            
            # Set stop loss and take profit
            order.stopLossLevel = stop_loss_price
            order.takeProfitLevel = price + (stop_loss_distance * 2.0)  # 2:1 risk-reward ratio
            
            self.evt.sendOrder(order)
            
            # Update state
            self.current_position = 1
            self.entry_price = price
            self.entry_time = current_time
            self.stop_loss_price = stop_loss_price
            self.take_profit_price = order.takeProfitLevel
            self.trailing_stop = price - (atr * self.trailing_stop_atr)
            self.last_signal_time = current_time
            
            self.evt.consoleLog(f"🚀 Open Long Position | Price: {price:.4f} | Size: {position_size} | Stop: {stop_loss_price:.4f}")
            
        except Exception as e:
            self.evt.consoleLog(f"Long position opening error: {str(e)}")

    def open_short_position(self, price, atr, current_time):
        """Open short position - Enhanced version"""
        try:
            # Calculate stop loss and position size
            stop_loss_distance = atr * self.atr_multiplier
            stop_loss_price = price + stop_loss_distance
            position_size = self.calculate_position_size(price, stop_loss_price)
            
            order = AlgoAPIUtil.OrderObject()
            order.instrument = self.myinstrument
            order.openclose = 'open'
            order.buysell = -1
            order.ordertype = 0
            order.volume = position_size
            
            # Set stop loss and take profit
            order.stopLossLevel = stop_loss_price
            order.takeProfitLevel = max(price - (stop_loss_distance * 2.0), 0.01)
            
            self.evt.sendOrder(order)
            
            # Update state
            self.current_position = -1
            self.entry_price = price
            self.entry_time = current_time
            self.stop_loss_price = stop_loss_price
            self.take_profit_price = order.takeProfitLevel
            self.trailing_stop = price + (atr * self.trailing_stop_atr)
            self.last_signal_time = current_time
            
            self.evt.consoleLog(f"🚀 Open Short Position | Price: {price:.4f} | Size: {position_size} | Stop: {stop_loss_price:.4f}")
            
        except Exception as e:
            self.evt.consoleLog(f"Short position opening error: {str(e)}")

    def check_exit_conditions(self, current_price, current_high, current_low, current_time):
        """Check exit conditions"""
        if self.current_position == 1:  # Long position
            self.update_trailing_stop(current_high, current_price)
            
            exit_condition = (
                current_price <= self.trailing_stop or
                current_price <= self.stop_loss_price or
                current_price >= self.take_profit_price or
                self.check_holding_time(current_time)
            )
            
            if exit_condition:
                self.close_long_position(current_price, current_time)
                
        elif self.current_position == -1:  # Short position
            self.update_trailing_stop(current_low, current_price)
            
            exit_condition = (
                current_price >= self.trailing_stop or
                current_price >= self.stop_loss_price or
                current_price <= self.take_profit_price or
                self.check_holding_time(current_time)
            )
            
            if exit_condition:
                self.close_short_position(current_price, current_time)

    def update_trailing_stop(self, extreme_price, current_price):
        """Update trailing stop"""
        if self.current_position == 1:  # Long
            new_trailing_stop = extreme_price - (self.trailing_stop_atr * self.get_current_atr())
            if new_trailing_stop > self.trailing_stop:
                self.trailing_stop = new_trailing_stop
                # Also update fixed stop loss
                self.stop_loss_price = max(self.stop_loss_price, new_trailing_stop)
                
        elif self.current_position == -1:  # Short
            new_trailing_stop = extreme_price + (self.trailing_stop_atr * self.get_current_atr())
            if new_trailing_stop < self.trailing_stop:
                self.trailing_stop = new_trailing_stop
                # Also update fixed stop loss
                self.stop_loss_price = min(self.stop_loss_price, new_trailing_stop)

    def get_current_atr(self):
        """Get current ATR value"""
        if len(self.close_prices) < self.atr_period:
            return 0
        try:
            high_array = np.array(self.high_prices[-self.atr_period:])
            low_array = np.array(self.low_prices[-self.atr_period:])
            close_array = np.array(self.close_prices[-self.atr_period:])
            atr = talib.ATR(high_array, low_array, close_array, timeperiod=self.atr_period)
            return atr[-1] if not np.isnan(atr[-1]) else 0
        except:
            return 0

    def close_long_position(self, current_price, current_time):
        """Close long position"""
        try:
            # Calculate P&L
            pnl = (current_price - self.entry_price) * self.get_position_value()
            self.record_trade_result(pnl > 0, pnl)
            
            order = AlgoAPIUtil.OrderObject()
            order.instrument = self.myinstrument
            order.openclose = 'open'
            order.buysell = -1
            order.ordertype = 0
            order.volume = self.get_position_value()  # Use actual position value
            
            self.evt.sendOrder(order)
            self.reset_position_state(current_time)
            
            status = "💰Profit" if pnl > 0 else "📉Loss"
            self.evt.consoleLog(f"📊 Close Long Position | Price: {current_price:.4f} | P&L: {pnl:.2f} ({status})")
            
        except Exception as e:
            self.evt.consoleLog(f"Long position closing error: {str(e)}")

    def close_short_position(self, current_price, current_time):
        """Close short position"""
        try:
            # Calculate P&L
            pnl = (self.entry_price - current_price) * self.get_position_value()
            self.record_trade_result(pnl > 0, pnl)
            
            order = AlgoAPIUtil.OrderObject()
            order.instrument = self.myinstrument
            order.openclose = 'open'
            order.buysell = 1
            order.ordertype = 0
            order.volume = self.get_position_value()  # Use actual position value
            
            self.evt.sendOrder(order)
            self.reset_position_state(current_time)
            
            status = "💰Profit" if pnl > 0 else "📉Loss"
            self.evt.consoleLog(f"📊 Close Short Position | Price: {current_price:.4f} | P&L: {pnl:.2f} ({status})")
            
        except Exception as e:
            self.evt.consoleLog(f"Short position closing error: {str(e)}")

    def get_position_value(self):
        """Get current position value"""
        # Need to calculate based on actual position, simplified return fixed value
        return self.base_position_size

    def reset_position_state(self, current_time):
        """Reset position state"""
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.trailing_stop = 0
        self.last_signal_time = current_time

    def close_all_positions(self, current_price, current_time, reason=""):
        """Force close all positions"""
        if self.current_position == 1:
            self.close_long_position(current_price, current_time)
            self.evt.consoleLog(f"⚠️ Force Close Long Position - {reason}")
        elif self.current_position == -1:
            self.close_short_position(current_price, current_time)
            self.evt.consoleLog(f"⚠️ Force Close Short Position - {reason}")

    def check_risk_limits(self):
        """Check risk limits"""
        # Check maximum drawdown
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.evt.consoleLog(f"⚠️ Maximum Drawdown Limit Reached: {current_drawdown:.2%}")
            return False
            
        # Check daily loss
        if self.daily_pnl < -self.max_daily_loss * self.current_equity:
            self.evt.consoleLog(f"⚠️ Daily Loss Limit Reached: {self.daily_pnl:.2f}")
            return False
            
        return True

    def record_trade_result(self, is_win, pnl):
        """Record trade result"""
        self.total_trades += 1
        if is_win:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        # Update equity curve
        self.current_equity += pnl
        self.daily_pnl += pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)
        
        # Output statistics every 10 trades
        if self.total_trades % 10 == 0:
            win_rate = (self.win_count / self.total_trades) * 100 if self.total_trades > 0 else 0
            profit_factor = abs(self.win_count * self.get_avg_win()) / abs(self.loss_count * self.get_avg_loss()) if self.loss_count > 0 else float('inf')
            self.evt.consoleLog(f"📈 Trading Statistics | Total: {self.total_trades} | Win Rate: {win_rate:.1f}% | Profit Factor: {profit_factor:.2f}")

    def get_avg_win(self):
        """Get average win (simplified)"""
        return self.base_position_size * 0.02  # Assume average 2% win

    def get_avg_loss(self):
        """Get average loss (simplified)"""
        return self.base_position_size * 0.01  # Assume average 1% loss

    def reset_daily_stats(self, current_time):
        """Reset daily statistics"""
        if self.last_reset_time is None or current_time.date() != self.last_reset_time.date():
            self.daily_pnl = 0
            self.last_reset_time = current_time

    def check_holding_time(self, current_time):
        """Check holding time"""
        if self.entry_time is None:
            return False
        holding_time = (current_time - self.entry_time).total_seconds()
        return holding_time > self.max_holding_period

    def on_marketdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        self.evt.consoleLog(f"📊 Daily P&L: {pl}")

    def on_openPositionfeed(self, op, oo, uo):
        pass