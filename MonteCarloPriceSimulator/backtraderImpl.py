import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
import datetime as dt
from MonteCarloSimulator import MonteCarloSimulator
import ccxt

# --- Backtrader Strategy Definition ---
class VolatilityBreakoutStrategy(bt.Strategy):
    """
    Backtrader Strategy implementing the Volatility Breakout logic.
    """
    params = (
        ('lookback_period', 96),
        ('profit_target', 0.01),
        ('stop_loss', 0.01),
        ('max_holding_period', 96),
        ('printlog', False), # Controls printing log messages
    )

    def __init__(self):
        # Keep references to data feeds
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_close = self.datas[0].close

        # Calculate rolling high/low range
        self.range_high = bt.indicators.Highest(self.data_high(-1), period=self.p.lookback_period)
        self.range_low = bt.indicators.Lowest(self.data_low(-1), period=self.p.lookback_period)

        # Order tracking
        self.order = None
        self.entry_price = None
        self.entry_bar = None # To track holding period

    def log(self, txt, dt=None, doprint=False):
        """Logging function for this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                self.entry_bar = len(self) # Record bar number at entry
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                self.entry_bar = len(self) # Record bar number at entry

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')


    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
             # Wait for enough bars for the lookback period
            if len(self) > self.p.lookback_period:
                # Check for long breakout
                if self.data_high[0] > self.range_high[0]:
                    self.log(f'LONG ENTRY SIGNAL: High {self.data_high[0]:.2f} > Range High {self.range_high[0]:.2f}')
                    # Calculate position size (e.g., use 95% of cash)
                    size = self.broker.get_cash() * 0.95 / self.data_close[0]
                    self.order = self.buy(size=size)
                    # Set stop loss and profit target prices upon entry confirmation in notify_order
                    # Or use bracket orders if supported / desired

                # Check for short breakout (if shorting is allowed/desired)
                # elif self.data_low[0] < self.range_low[0]:
                #     self.log(f'SHORT ENTRY SIGNAL: Low {self.data_low[0]:.2f} < Range Low {self.range_low[0]:.2f}')
                #     size = self.broker.get_cash() * 0.95 / self.data_close[0] # Adjust sizing for short
                #     self.order = self.sell(size=size)
        
        else: # Already in the market, check exit conditions
            current_bar = len(self)
            holding_period = current_bar - self.entry_bar
            
            exit_signal = False
            exit_reason = ""

            # Check Profit Target
            if self.position.size > 0: # Long position
                target_price = self.entry_price * (1 + self.p.profit_target)
                stop_price = self.entry_price * (1 - self.p.stop_loss)
                if self.data_close[0] >= target_price:
                    exit_signal = True
                    exit_reason = "Profit Target"
                elif self.data_close[0] <= stop_price:
                    exit_signal = True
                    exit_reason = "Stop Loss"
            
            # Add Short position exit logic if shorting is enabled
            # elif self.position.size < 0: # Short position
            #     target_price = self.entry_price * (1 - self.p.profit_target)
            #     stop_price = self.entry_price * (1 + self.p.stop_loss)
            #     if self.data_close[0] <= target_price:
            #         exit_signal = True
            #         exit_reason = "Profit Target"
            #     elif self.data_close[0] >= stop_price:
            #         exit_signal = True
            #         exit_reason = "Stop Loss"

            # Check Max Holding Period
            if not exit_signal and holding_period >= self.p.max_holding_period:
                exit_signal = True
                exit_reason = "Max Holding Period"

            # Execute exit if signal triggered
            if exit_signal:
                self.log(f'EXIT SIGNAL: {exit_reason}. Closing position.')
                self.order = self.close() # Close the current position


# --- Backtrader Simulator Class ---
class BacktraderVolatilityBreakoutSimulator(MonteCarloSimulator):
    def __init__(self, historical_data: pd.DataFrame, timeframe: str, num_simulations: int,
                 simulation_length: int, random_seed: int = None,
                 lookback_period: int = 96, profit_target: float = 0.01, stop_loss: float = 0.01,
                 max_holding_period: int = 96, initial_cash: float = 10000.0, commission: float = 0.001):
        """Initialize the Backtrader Volatility Breakout simulator.

        Args:
            historical_data: DataFrame with OHLCV columns.
            timeframe: String like '15m', '1h', '1d'.
            num_simulations: Number of price paths to simulate and backtest.
            simulation_length: Number of periods to simulate in each path.
            random_seed: Optional seed for reproducibility.
            lookback_period: Bars for breakout range (default: 96).
            profit_target: Profit target fraction (default: 0.01).
            stop_loss: Stop loss fraction (default: 0.01).
            max_holding_period: Max bars to hold a position (default: 96).
            initial_cash: Starting capital for backtrader (default: 10000.0).
            commission: Broker commission per trade (default: 0.001 = 0.1%).
        """
        super().__init__(historical_data, timeframe, num_simulations, simulation_length, random_seed)

        # Strategy & Backtrader parameters
        self.lookback_period = lookback_period
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        self.initial_cash = initial_cash
        self.commission = commission
        
        self.results = [] # To store results from each simulation run
        self.aggregated_results = None
        self.all_equity_curves = []

    def _get_time_frequency(self):
        """ Converts timeframe string to pandas frequency string. """
        tf_map = {'1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T', 
                  '1h': 'H', '4h': '4H', '1d': 'D'}
        if self.timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe for frequency conversion: {self.timeframe}")
        return tf_map[self.timeframe]

    def run_single_backtest(self, path_data: np.ndarray):
        """Runs a backtrader backtest on a single simulated price path."""
        
        # 1. Create Cerebro engine
        cerebro = bt.Cerebro(stdstats=False) # Disable standard observers initially

        # 2. Prepare Data Feed
        # Create a dummy datetime index for the simulated data
        start_date = dt.datetime(2025, 3, 22) # Arbitrary start date
        freq = self._get_time_frequency()
        datetime_index = pd.date_range(start=start_date, periods=self.simulation_length, freq=freq)
        
        # Create DataFrame compatible with PandasData feed
        df = pd.DataFrame({
            'datetime': datetime_index,
            'open': path_data[:, 0],
            'high': path_data[:, 1],
            'low': path_data[:, 2],
            'close': path_data[:, 3],
            #'volume': np.random.randint(100, 1000, size=self.simulation_length) # Dummy volume
        })
        df.set_index('datetime', inplace=True)
        #df['openinterest'] = 0 # Required column, set to 0

        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)

        # 3. Add Strategy
        cerebro.addstrategy(
            VolatilityBreakoutStrategy,
            lookback_period=self.lookback_period,
            profit_target=self.profit_target,
            stop_loss=self.stop_loss,
            max_holding_period=self.max_holding_period,
            printlog=False # Keep logs off for bulk runs
        )

        # 4. Set Initial Capital and Commission
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # 5. Add Analyzers for Performance Tracking
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0) # Adjust as needed
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn') # To get equity curve data

        # 6. Run the Backtest
        # Use runonce() if analyzers need daily updates, run() is usually fine
        strategy_instance = cerebro.run()[0] 

        # 7. Extract Results from Analyzers
        analyzers = strategy_instance.analyzers
        run_results = {}
        
        # Basic Returns and Drawdown
        run_results['final_value'] = cerebro.broker.getvalue()
        run_results['total_return'] = analyzers.returns.get_analysis().get('rtot', 0.0)
        run_results['max_drawdown'] = analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
        
        # Trade Analysis
        trade_analysis = analyzers.tradeanalyzer.get_analysis()
        run_results['num_trades'] = trade_analysis.get('total', {}).get('total', 0)
        run_results['win_rate'] = 0.0
        run_results['profit_factor'] = float('inf')
        if run_results['num_trades'] > 0:
             wins = trade_analysis.get('won', {}).get('total', 0)
             losses = trade_analysis.get('lost', {}).get('total', 0)
             if run_results['num_trades'] > 0:
                run_results['win_rate'] = wins / run_results['num_trades'] if run_results['num_trades'] > 0 else 0.0

             gross_profit = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0.0)
             gross_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0.0))
             run_results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Other metrics
        run_results['sharpe_ratio'] = analyzers.sharpe.get_analysis().get('sharperatio', None) # May be None if few trades
        run_results['sqn'] = analyzers.sqn.get_analysis().get('sqn', None)
        
        # Equity Curve Data (as TimeReturn dictionary)
        time_return_analysis = analyzers.timereturn.get_analysis()
        # Convert TimeReturn dict to a simple list/array of portfolio values over time
        # The keys are datetime objects, values are returns for that period. We need cumulative value.
        equity_curve = pd.Series(time_return_analysis).add(1).cumprod() * self.initial_cash
        run_results['equity_curve'] = equity_curve # Store as Pandas Series

        return run_results

    def backtest_strategy(self):
        """Runs the backtrader backtest on all generated Monte Carlo paths."""
        if self.simulated_paths is None:
            print("No simulated paths available. Generating paths first...")
            # Ensure OHLC paths are generated
            self.generate_price_paths(model_type="GBM", close_only=False) 
        
        if self.simulated_paths.shape[2] != 4:
             raise ValueError("Backtrader simulation requires OHLC paths. Generate paths with close_only=False.")

        print(f"Running backtrader backtest on {self.num_simulations} simulated paths...")
        self.results = []
        self.all_equity_curves = []

        for i in range(self.num_simulations):
            path_data = self.simulated_paths[i]
            try:
                single_run_result = self.run_single_backtest(path_data)
                self.results.append(single_run_result)
                # Store equity curve separately for easier plotting later
                self.all_equity_curves.append(single_run_result['equity_curve']) 
            except Exception as e:
                print(f"Error backtesting path {i}: {e}")
                # Optionally append None or skip to handle errors gracefully
                self.results.append(None) 
                self.all_equity_curves.append(None)
            
            if (i + 1) % (self.num_simulations // 10 or 1) == 0:
                 print(f"Completed {i + 1}/{self.num_simulations} simulations...")

        print("Backtesting complete. Aggregating results...")
        self._aggregate_results()
        return self.aggregated_results

    def _aggregate_results(self):
        """Aggregates metrics from all individual simulation backtests."""
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            print("No valid backtest results found.")
            self.aggregated_results = {}
            return

        agg = {}
        metrics_to_average = ['total_return', 'max_drawdown', 'num_trades', 'win_rate', 'profit_factor', 'sharpe_ratio', 'sqn']
        
        for metric in metrics_to_average:
             # Filter out None or inf values before averaging for certain metrics
            valid_values = [r[metric] for r in valid_results if r.get(metric) is not None and np.isfinite(r.get(metric))]
            if valid_values:
                agg[f'avg_{metric}'] = np.mean(valid_values)
                agg[f'median_{metric}'] = np.median(valid_values)
            else:
                agg[f'avg_{metric}'] = None
                agg[f'median_{metric}'] = None


        # Special handling for returns
        total_returns = [r['total_return'] for r in valid_results if 'total_return' in r]
        if total_returns:
             agg['avg_total_return'] = np.mean(total_returns)
             agg['median_total_return'] = np.median(total_returns)
             agg['std_total_return'] = np.std(total_returns)
             agg['best_return'] = np.max(total_returns)
             agg['worst_return'] = np.min(total_returns)
             agg['return_quantiles'] = np.percentile(total_returns, [5, 25, 50, 75, 95])

        # Final Values
        final_values = [r['final_value'] for r in valid_results if 'final_value' in r]
        if final_values:
            agg['avg_final_value'] = np.mean(final_values)
            agg['median_final_value'] = np.median(final_values)
            agg['std_final_value'] = np.std(final_values)


        self.aggregated_results = agg
        print("Result aggregation complete.")

    def plot_backtrader_equity_curves(self, n_curves=20, title="Backtrader Strategy Equity Curves"):
        """Plots a subset of equity curves from the backtrader results.

        Args:
            n_curves: Number of individual curves to plot (default 20).
            title: Plot title.
        """
        valid_curves = [ec for ec in self.all_equity_curves if ec is not None and not ec.empty]
        
        if not valid_curves:
            print("No valid equity curves available to plot.")
            return

        plt.figure(figsize=(14, 7))

        # Determine how many curves to plot
        n_to_plot = min(n_curves, len(valid_curves))

        # Randomly sample curves to plot
        indices = np.random.choice(len(valid_curves), n_to_plot, replace=False)

        # Plot each sampled equity curve
        for i, idx in enumerate(indices):
            # Use the equity curve's index (datetime) for the x-axis
            plt.plot(valid_curves[idx].index, valid_curves[idx].values, alpha=0.4, label=f"Path {idx+1}" if n_curves < 30 else None) # Avoid overcrowding legend

        # Calculate and plot the mean equity curve
        # Need to align curves by index (time) before averaging
        try:
            all_curves_df = pd.concat(valid_curves, axis=1) # Aligns by datetime index
            mean_equity = all_curves_df.mean(axis=1)
            plt.plot(mean_equity.index, mean_equity.values, color='black', linewidth=2.5, label="Mean Equity")
        except Exception as e:
            print(f"Could not calculate or plot mean equity curve: {e}")
            # Fallback: Plot mean of final values if alignment fails
            if self.aggregated_results and 'avg_final_value' in self.aggregated_results:
                 plt.axhline(self.aggregated_results['avg_final_value'], color='red', linestyle='--', label=f"Avg Final Value ({self.aggregated_results['avg_final_value']:.2f})")


        # Add formatting
        plt.title(title)
        plt.xlabel("Date / Time")
        plt.ylabel(f"Portfolio Value (Initial = {self.initial_cash})")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        # Improve date formatting on x-axis if needed
        plt.gcf().autofmt_xdate() 
        plt.show()

# --- Helper functions (can be kept separate or moved into the class) ---
def fetch_historical_data(symbol, timeframe, limit=500):
    try:
        exchangeQ = ccxt.mexc({
           'apiKey': '',#
            'secret': '',
            "enableRateLimit": True,
            'options': {
                'recvWindow': 20000
            }})
        all_ohlcv = []
        since = None  # Start from the most recent candle

        while len(all_ohlcv) < limit:
            # Fetch up to 500 candles per request
            ohlcv = exchangeQ.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
            if not ohlcv:
                break  # No more data available

            all_ohlcv.extend(ohlcv)
            # Move backward in time by subtracting the duration of the fetched candles
            since = ohlcv[0][0] - (exchangeQ.parse_timeframe(timeframe) * 1000)

        # Convert to DataFrame
        data = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        
        # Ensure the data is sorted by timestamp in ascending order (oldest first)
        data = data.sort_values(by="timestamp")
        
        # Return the most recent `limit` candles
        return data.iloc[-limit:]

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def plot_original_data(data):
    if data is None or data.empty:
        print("No data to plot.")
        return
    # Plot the line chart for closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(data["timestamp"], data["close"], label="Close Price", color="blue", linewidth=1.5)
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.title("Historical Close Price")
    plt.legend()
    plt.grid()
    plt.show()

# --- Example Flow using the new Backtrader Simulator ---
if __name__ == "__main__":
    # 1. Fetch historical data
    # Using a longer history for better simulation basis
    historical_data = fetch_historical_data("BTC/USDT", "15m", limit=10000) 
    if historical_data is None:
        print("Failed to fetch data. Exiting.")
        exit()
        
    #plot_original_data(historical_data) # Optional: visualize input data

    # 2. Initialize the Backtrader Simulator
    bt_sim = BacktraderVolatilityBreakoutSimulator(
        historical_data=historical_data,
        timeframe="15m",           # Match fetched data timeframe
        num_simulations=100,     # Reduced for faster example run
        simulation_length=10000,   # Length of each simulated future path
        random_seed=42,
        lookback_period=192,     # e.g., 24 hours for 1h timeframe
        profit_target=0.03,     # 3% TP
        stop_loss=0.01,       # 1.5% SL
        max_holding_period=48, # Max 2 days hold
        initial_cash=10000.0,
        commission=0.0005        # 0.05% commission
    )

    # 3. Generate price paths (using the base class method)
    # Important: Ensure OHLC paths are generated
    bt_sim.generate_price_paths(model_type="GBM") 
    # Optional: Plot some generated paths
    bt_sim.plot_simulated_paths(num_paths_to_plot=20) 

    # 4. Run backtests on all paths using backtrader
    results = bt_sim.backtest_strategy()

    # 5. Print Aggregated Results
    print("\n--- Aggregated Backtrader Results ---")
    if results:
        for key, value in results.items():
            if isinstance(value, (int, float)):
                 print(f"  {key}: {value:.4f}")
            elif isinstance(value, np.ndarray):
                 # Print quantiles nicely
                 if 'quantiles' in key:
                     print(f"  {key}: {np.round(value, 4).tolist()}")
                 else:
                     print(f"  {key}: [Array data, length {len(value)}]")
            else:
                 print(f"  {key}: {value}")
    else:
        print("No results to display.")

    # 6. Plot Equity Curves
    bt_sim.plot_backtrader_equity_curves(n_curves=25, title="Volatility Breakout - Backtrader Equity Curves")