import pandas as pd
import numpy as np
from MonteCarloSimulator import MonteCarloSimulator  # Import the class directly
import ccxt
import matplotlib.pyplot as plt

class VolatilityBreakoutSimulator(MonteCarloSimulator):
    def __init__(self, historical_data: pd.DataFrame, timeframe: str, num_simulations: int, 
                 simulation_length: int, random_seed: int = None, 
                 lookback_period: int = 96, profit_target: float = 0.01, stop_loss: float = 0.01, 
                 max_holding_period: int = 96):
        """Initialize the Volatility Breakout simulator with strategy parameters.

        Args:
            historical_data: DataFrame with OHLCV columns.
            timeframe: String like '15m', '1h', '1d'.
            num_simulations: Number of price paths.
            simulation_length: Number of periods to simulate.
            random_seed: Optional seed for reproducibility.
            lookback_period: Bars to calculate breakout range (default: 96 = 24h at 15m).
            profit_target: Profit target as a fraction (default: 0.01 = 1%).
            stop_loss: Stop loss as a fraction (default: 0.01 = 1%).
            max_holding_period: Max bars to hold a position (default: 96 = 24h).
        """
        super().__init__(historical_data, timeframe, num_simulations, simulation_length, random_seed)
        
        # Strategy parameters
        self.lookback_period = lookback_period
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period

    def calculate_indicators(self, indicator_list: list = None):
        """Calculate the 96-bar rolling high/low range for breakout detection.

        Returns:
            dict: Contains 'range_high' and 'range_low' arrays for each path.
        """
        if self.simulated_paths is None:
            raise ValueError("No simulated paths available. Run generate_price_paths first.")
        
        # Ensure OHLC paths are available
        if self.simulated_paths.shape[2] != 4:
            raise ValueError("Volatility Breakout requires OHLC paths, not close_only")

        # Extract high and low from simulated paths
        highs = self.simulated_paths[:, :, 1]  # High prices
        lows = self.simulated_paths[:, :, 2]   # Low prices
        
        # Initialize arrays for rolling ranges
        range_high = np.zeros((self.num_simulations, self.simulation_length))
        range_low = np.zeros((self.num_simulations, self.simulation_length))
        
        # Calculate rolling max/min over lookback_period
        for t in range(self.lookback_period, self.simulation_length):
            range_high[:, t] = np.max(highs[:, t-self.lookback_period:t], axis=1)
            range_low[:, t] = np.min(lows[:, t-self.lookback_period:t], axis=1)
        
        # Initialize the first lookback_period with expanding windows
        for t in range(self.lookback_period):
            if t > 0:  # For t=0, keep zeros as there's no history
                range_high[:, t] = np.max(highs[:, :t+1], axis=1)
                range_low[:, t] = np.min(lows[:, :t+1], axis=1)
        
        return {
            "range_high": range_high,
            "range_low": range_low
        }

 
    def backtest_strategy(self):
        """Backtest the Volatility Breakout strategy on simulated paths.

        Returns:
            dict: Performance metrics (total_return, win_rate, max_drawdown, num_trades, profit_factor).
        """
        if self.simulated_paths is None:
            raise ValueError("No simulated paths available. Run generate_price_paths first.")
        
        # Generate ranges
        indicators = self.calculate_indicators()
        range_high = indicators["range_high"]
        range_low = indicators["range_low"]
        
        # Extract OHLC from paths
        opens = self.simulated_paths[:, :, 0]
        highs = self.simulated_paths[:, :, 1]
        lows = self.simulated_paths[:, :, 2]
        closes = self.simulated_paths[:, :, 3]
        
        # Initialize tracking arrays
        total_returns = np.zeros(self.num_simulations)
        num_trades = np.zeros(self.num_simulations)
        win_counts = np.zeros(self.num_simulations)
        equity_curves = np.ones((self.num_simulations, self.simulation_length))  # Initialize all to 1.0
        
        # Backtest each path
        for path_idx in range(self.num_simulations):
            in_position = False
            entry_price = 0.0
            entry_time = 0
            equity = 1.0
            
            for t in range(self.lookback_period, self.simulation_length):
                # Only update equity curve if we've made a trade
                equity_curves[path_idx, t] = equity
                
                if not in_position:
                    # Check for breakout
                    if highs[path_idx, t] > range_high[path_idx, t]:
                        # Long entry
                        entry_price = closes[path_idx, t]
                        entry_time = t
                        in_position = True
                        num_trades[path_idx] += 1
                    elif lows[path_idx, t] < range_low[path_idx, t]:
                        # Short entry
                        entry_price = closes[path_idx, t]
                        entry_time = t
                        in_position = True
                        num_trades[path_idx] += 1
                else:
                    # Check exit conditions
                    current_price = closes[path_idx, t]
                    holding_period = t - entry_time
                    
                    # Calculate profit based on position type (assuming long position for simplicity)
                    profit = (current_price - entry_price) / entry_price
                    
                    # Exit conditions with risk management
                    if profit >= self.profit_target:
                        # Take profit exit
                        equity *= (1 + profit)
                        in_position = False
                        win_counts[path_idx] += 1
                    elif profit <= -self.stop_loss:
                        # Stop loss exit - limit loss to stop_loss percentage
                        equity *= (1 - self.stop_loss)  # Cap the loss at stop loss level
                        in_position = False
                    elif holding_period >= self.max_holding_period:
                        # Time-based exit
                        equity *= (1 + profit)
                        in_position = False
                        if profit > 0:
                            win_counts[path_idx] += 1
                
                # Update equity curve
                equity_curves[path_idx, t] = equity
            
            # Close any open positions at the end of simulation
            if in_position:
                profit = (closes[path_idx, -1] - entry_price) / entry_price
                equity *= (1 + profit)
                if profit > 0:
                    win_counts[path_idx] += 1
            
            total_returns[path_idx] = equity - 1.0  # Final return
        
        # Fill all equity curve values that weren't updated during backtest
        for path_idx in range(self.num_simulations):
            for t in range(1, self.simulation_length):
                if equity_curves[path_idx, t] == 1.0 and equity_curves[path_idx, t-1] != 1.0:
                    equity_curves[path_idx, t] = equity_curves[path_idx, t-1]
        
        # Calculate aggregate metrics
        avg_total_return = np.mean(total_returns)
        median_total_return = np.median(total_returns)
        avg_num_trades = np.mean(num_trades)
        
        # Calculate win rate (avoid division by zero)
        win_rate = np.mean(win_counts / np.maximum(num_trades, 1)) 
        
        # Calculate max drawdown properly
        max_drawdowns = np.zeros(self.num_simulations)
        for path_idx in range(self.num_simulations):
            peak = np.maximum.accumulate(equity_curves[path_idx])
            drawdown = (peak - equity_curves[path_idx]) / peak
            max_drawdowns[path_idx] = np.max(drawdown)
        avg_max_drawdown = np.mean(max_drawdowns)
        
        # Profit factor (gross profit / gross loss)
        profits = total_returns[total_returns > 0].sum()
        losses = abs(total_returns[total_returns < 0]).sum()
        profit_factor = profits / losses if losses > 0 else float("inf")
        
        # Best and worst case scenarios
        best_return = np.max(total_returns)
        worst_return = np.min(total_returns)
        
        return {
            "total_return": avg_total_return,
            "median_return": median_total_return,
            "win_rate": win_rate,
            "max_drawdown": avg_max_drawdown,
            "num_trades": avg_num_trades,
            "profit_factor": profit_factor,
            "best_return": best_return,
            "worst_return": worst_return,
            "equity_curves": equity_curves
        }

def plot_original_data(data):
    # Plot the line chart for closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(data["timestamp"], data["close"], label="Close Price", color="blue", linewidth=1.5)

    # Formatting
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.title("OHLC Line Chart (Close Price)")
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

def fetch_historical_data(symbol, timeframe, limit=500):
    try:
        exchangeQ = ccxt.mexc({
            'apiKey': '',#mx0vglvNw0UoaHMgop
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

def plot_equity_curves(equity_curves, n_curves=20, title="Strategy Equity Curves"):
    """Plot a subset of equity curves from backtest results.
    
    Args:
        equity_curves: Array of equity curves (n_sim x n_periods)
        n_curves: Number of curves to plot (default 20)
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Determine how many curves to plot (min of n_curves or available curves)
    n_to_plot = min(n_curves, equity_curves.shape[0])
    
    # Randomly sample curves to plot
    indices = np.random.choice(equity_curves.shape[0], n_to_plot, replace=False)
    
    # Plot each sampled equity curve
    for i, idx in enumerate(indices):
        plt.plot(equity_curves[idx], alpha=0.5, label=f"Path {idx+1}")
    
    # Plot the mean equity curve
    mean_equity = np.mean(equity_curves, axis=0)
    plt.plot(mean_equity, color='black', linewidth=2, label="Mean Equity")
    
    # Add formatting
    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel("Equity (starting=1.0)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Example flow
if __name__ == "__main__":
    # Fetch historical data
    historical_data = fetch_historical_data("BTC/USDT:USDT", "15m", 1920)
    plot_original_data(historical_data)
    
    # Initialize and run
    sim = VolatilityBreakoutSimulator(
        historical_data=historical_data,
        timeframe="15m",
        num_simulations=1000,
        simulation_length=1920,
        random_seed=42,
        profit_target=0.02,    # Increased to 2%
        stop_loss=0.02,        # Kept at 1%
        lookback_period=48     # Reduced from 96 to 48 periods (12 hours)
    )
    
    # Generate price paths
    sim.generate_price_paths(model_type="GARCH+JUMP")
    sim.plot_simulated_paths(num_paths_to_plot=25)
    
    # Run backtest and show results
    results = sim.backtest_strategy()
    print("Backtest Results:")
    for key, value in results.items():
        if key != "equity_curves":
            print(f"  {key}: {value}")
    
    # Plot equity curves
    plot_equity_curves(results["equity_curves"], n_curves=25, 
                      title="Volatility Breakout Strategy - Equity Curves")