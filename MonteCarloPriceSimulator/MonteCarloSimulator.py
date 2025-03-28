import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from arch import arch_model  # For GARCH fitting
"""
Improvemtents for future:
Correlation modeling for multiple assets
Seasonal effects (time of day, day of week patterns)
Modeling of liquidity constraints and slippage
Additional market regimes (e.g., trending vs ranging)
"""
class MonteCarloSimulator(ABC):
    def __init__(self, historical_data: pd.DataFrame, timeframe: str, num_simulations: int, 
                 simulation_length: int, random_seed: int = None):
        """Initialize the Monte Carlo simulator with historical data and simulation parameters.

        Args:
            historical_data: DataFrame with OHLCV columns (at least 'close').
            timeframe: String like '15m', '1h', '1d' specifying data granularity.
            num_simulations: Number of price paths to generate.
            simulation_length: Number of periods to simulate.
            random_seed: Optional seed for reproducibility.
        """
        # Input validation
        if not isinstance(historical_data, pd.DataFrame) or "close" not in historical_data.columns:
            raise ValueError("historical_data must be a DataFrame with 'close' column")
        if not isinstance(num_simulations, int) or num_simulations <= 0:
            raise ValueError("num_simulations must be a positive integer")
        if not isinstance(simulation_length, int) or simulation_length <= 0:
            raise ValueError("simulation_length must be a positive integer")
        if not isinstance(timeframe, str) or timeframe not in ["15m", "1h", "1d"]:
            raise ValueError("timeframe must be '15m', '1h', or '1d'")
        if len(historical_data) < simulation_length:
            raise ValueError("simulation_length exceeds available historical data")

        # Store inputs
        self.historical_data = historical_data
        self.timeframe = timeframe
        self.num_simulations = num_simulations
        self.simulation_length = simulation_length
        self.random_seed = random_seed

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize simulation variables
        self.simulated_paths = None
        self.model_params = {}

        # Calculate statistics and fit models
        self._calculate_statistics()
        self._fit_garch_model()
        self._estimate_jump_params()

    def _calculate_statistics(self):
        """Compute statistics including market regimes."""
        returns = np.log(self.historical_data["close"] / self.historical_data["close"].shift(1)).dropna()
        if len(returns) < 2:
            raise ValueError("Insufficient data to calculate statistics")
            
        # Basic statistics
        self.mu = returns.mean()
        self.sigma = returns.std()
        
        print("\nDebug Statistics:")
        print(f"Mean return (mu): {self.mu:.6f}")
        print(f"Return std (sigma): {self.sigma:.6f}")
        print(f"Annualized Volatility: {self.sigma * np.sqrt(365*24*4):.2f}")  # For 15min data
        
        # Detect market regimes using rolling statistics
        window = min(60, len(returns) // 4)  # 15-hour window for 15m data
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        # Classify regimes
        self.regimes = {
            'bullish': {
                'mu': np.mean(returns[rolling_mean > 0]),
                'sigma': np.std(returns[rolling_mean > 0])
            },
            'bearish': {
                'mu': np.mean(returns[rolling_mean < 0]),
                'sigma': np.std(returns[rolling_mean < 0])
            },
            'high_vol': {
                'mu': np.mean(returns[rolling_std > self.sigma]),
                'sigma': np.std(returns[rolling_std > self.sigma])
            },
            'low_vol': {
                'mu': np.mean(returns[rolling_std < self.sigma]),
                'sigma': np.std(returns[rolling_std < self.sigma])
            }
        }
        
        print("\nRegime Parameters:")
        for regime, params in self.regimes.items():
            print(f"{regime}: mu={params['mu']:.6f}, sigma={params['sigma']:.6f}")
        
        # Calculate mean reversion parameters
        price_series = np.log(self.historical_data["close"])
        price_diff = price_series - price_series.rolling(window=window).mean()
        price_diff = price_diff.dropna()
        
        if len(price_diff) > 1:
            self.mean_reversion_speed = -np.polyfit(price_diff[:-1], np.diff(price_diff), 1)[0]
            # Reduce mean reversion effect significantly
            self.mean_reversion_speed = min(self.mean_reversion_speed * 0.1, 0.01)
            print(f"\nMean Reversion Speed: {self.mean_reversion_speed:.6f}")
        else:
            self.mean_reversion_speed = 0.001
            print("\nUsing default mean reversion speed: 0.001")

    def _fit_garch_model(self):
        """Fit GARCH(1,1) model to historical returns."""
        returns = np.log(self.historical_data["close"] / self.historical_data["close"].shift(1)).dropna()
        model = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero", dist="Normal")
        result = model.fit(disp="off")
        self.model_params["garch"] = {
            "omega": result.params["omega"],
            "alpha": result.params["alpha[1]"],
            "beta": result.params["beta[1]"]
        }
        
        print("\nGARCH Parameters:")
        print(f"omega: {self.model_params['garch']['omega']:.6f}")
        print(f"alpha: {self.model_params['garch']['alpha']:.6f}")
        print(f"beta: {self.model_params['garch']['beta']:.6f}")

    def _estimate_jump_params(self):
        """Estimate jump frequency and size from historical data."""
        returns = np.log(self.historical_data["close"] / self.historical_data["close"].shift(1)).dropna()
        jump_threshold = 2 * self.sigma  # Reduced from 3σ to 2σ for more frequent jumps
        jumps = returns[abs(returns) > jump_threshold]
        jump_freq = len(jumps) / len(returns)
        jump_std = jumps.std() if len(jumps) > 0 else 0.02
        
        self.model_params["jump"] = {
            "lambda": min(max(jump_freq, 0.005), 0.02),  # Increased frequency range
            "jump_mean": 0,
            "jump_std": min(jump_std, 0.05)  # Increased max jump size
        }
        
        print("\nJump Parameters:")
        print(f"lambda (frequency): {self.model_params['jump']['lambda']:.6f}")
        print(f"jump_std: {self.model_params['jump']['jump_std']:.6f}")
        print(f"Number of detected jumps: {len(jumps)}")

    def generate_price_paths(self, model_type: str = "GBM", return_type: str = "ohlc", 
                            model_params: dict = None):
        """Generate simulated price paths using simplified Geometric Brownian Motion.
        
        This is a greatly simplified approach that creates more realistic paths
        by using the historical data's characteristics without complex modeling.
        """
        # Use LAST price as starting point (fix: was using first price)
        start_close = self.historical_data["close"].iloc[-1]
        start_open = self.historical_data.get("open", start_close).iloc[-1]
        start_high = self.historical_data.get("high", start_close).iloc[-1]
        start_low = self.historical_data.get("low", start_close).iloc[-1]
        
        print(f"\nStarting simulation from last historical price: {start_close:.2f}")
        
        # Get historical returns for statistical properties
        hist_returns = np.log(self.historical_data["close"] / self.historical_data["close"].shift(1)).dropna().values
        
        # Calculate volatility statistics - now with asymmetry
        volatility = np.std(hist_returns)
        
        # Calculate asymmetric volatility (up vs down moves)
        up_returns = hist_returns[hist_returns > 0]
        down_returns = hist_returns[hist_returns < 0]
        up_vol = np.std(up_returns) if len(up_returns) > 0 else volatility
        down_vol = np.std(down_returns) if len(down_returns) > 0 else volatility
        
        # Calculate skewness for more realistic distributions
        skew = np.mean((hist_returns - np.mean(hist_returns))**3) / (volatility**3)
        
        print(f"Historical statistics:")
        print(f"  Mean return: {np.mean(hist_returns):.6f}")
        print(f"  Overall volatility: {volatility:.6f}")
        print(f"  Upside volatility: {up_vol:.6f}")
        print(f"  Downside volatility: {down_vol:.6f}")
        print(f"  Skewness: {skew:.4f}")
        
        # Initialize arrays for simulated prices
        closes = np.zeros((self.num_simulations, self.simulation_length))
        closes[:, 0] = start_close
        
        # Generate paths with asymmetric volatility
        for t in range(1, self.simulation_length):
            # Generate random shocks with slight skew
            z = np.random.normal(0, 1, self.num_simulations)
            
            # Apply different volatility for up and down moves
            returns = np.zeros(self.num_simulations)
            
            # Determine if move is up or down and apply appropriate volatility
            up_moves = z > 0
            down_moves = ~up_moves
            
            # Apply asymmetric volatility
            returns[up_moves] = z[up_moves] * up_vol
            returns[down_moves] = z[down_moves] * down_vol
            
            # Add a small drift component based on historical average
            drift = np.mean(hist_returns)
            returns += drift
            
            # Update prices
            closes[:, t] = closes[:, t-1] * np.exp(returns)
        
        # Generate OHLC data if requested
        if return_type == "ohlc":
            paths = np.zeros((self.num_simulations, self.simulation_length, 4))
            paths[:, 0, 0] = start_open  # Open
            paths[:, 0, 1] = start_high  # High
            paths[:, 0, 2] = start_low   # Low
            paths[:, 0, 3] = start_close # Close
            
            for t in range(1, self.simulation_length):
                # Close price
                paths[:, t, 3] = closes[:, t]
                # Open price (previous close)
                paths[:, t, 0] = closes[:, t-1]
                
                # Calculate returns for this bar
                r_t = np.log(closes[:, t] / closes[:, t-1])
                
                # Asymmetric high/low calculation
                up_bars = r_t > 0
                down_bars = ~up_bars
                
                # Initialize high/low at close prices
                paths[:, t, 1] = closes[:, t]  # High
                paths[:, t, 2] = closes[:, t]  # Low
                
                # For up bars: high further from close than low
                high_range_up = 0.7 * up_vol * np.abs(z[up_bars])
                low_range_up = 0.3 * up_vol * np.abs(z[up_bars])
                
                # For down bars: low further from close than high
                high_range_down = 0.3 * down_vol * np.abs(z[down_bars])
                low_range_down = 0.7 * down_vol * np.abs(z[down_bars])
                
                # Apply the ranges
                paths[up_bars, t, 1] = closes[up_bars, t] * np.exp(high_range_up)  # Higher high for up bars
                paths[up_bars, t, 2] = closes[up_bars, t] * np.exp(-low_range_up)  # Not so low
                
                paths[down_bars, t, 1] = closes[down_bars, t] * np.exp(high_range_down)  # Not so high
                paths[down_bars, t, 2] = closes[down_bars, t] * np.exp(-low_range_down)  # Lower low for down bars
                
                # Ensure proper ordering (high >= open/close >= low)
                paths[:, t, 1] = np.maximum(paths[:, t, 1], 
                                           np.maximum(paths[:, t, 0], paths[:, t, 3]))
                paths[:, t, 2] = np.minimum(paths[:, t, 2], 
                                           np.minimum(paths[:, t, 0], paths[:, t, 3]))
            
            self.simulated_paths = paths
        else:
            # Just close prices
            self.simulated_paths = closes[:, :, np.newaxis]
        
        print("\nSimulation complete:")
        print(f"  Starting price: {start_close:.2f}")
        print(f"  Number of paths: {self.num_simulations}")
        print(f"  Simulation length: {self.simulation_length} periods")
        
        return self.simulated_paths

    def plot_simulated_paths(self, num_paths_to_plot: int = 25, price_type: str = "close", 
                         show_historical: bool = True, show_percentiles: bool = True):
        """Plot simulated price paths with enhanced visualization.

        Args:
            num_paths_to_plot: Number of paths to plot (default: 25).
            price_type: 'open', 'high', 'low', or 'close' (default: 'close').
            show_historical: Whether to overlay historical close prices (default: True).
            show_percentiles: Whether to display confidence intervals (default: True).
        """
        if self.simulated_paths is None:
            raise ValueError("No simulated paths available. Run generate_price_paths first.")
        
        # Limit number of paths to plot
        num_to_plot = min(num_paths_to_plot, self.num_simulations)
        
        # Determine price index based on type
        price_idx = {"open": 0, "high": 1, "low": 2, "close": 3}.get(price_type, 3)
        
        plt.figure(figsize=(12, 6))
        
        # Extract prices from simulated paths
        if self.simulated_paths.shape[2] == 1:  # close_only
            sim_prices = self.simulated_paths[:, :, 0]
        else:  # ohlc
            sim_prices = self.simulated_paths[:, :, price_idx]
        
        # Calculate price range for y-axis limits
        min_price = np.min(sim_prices)
        max_price = np.max(sim_prices)
        
        # Calculate percentiles for confidence intervals
        if show_percentiles and self.num_simulations >= 10:
            p10 = np.percentile(sim_prices, 10, axis=0)
            p25 = np.percentile(sim_prices, 25, axis=0)
            p50 = np.percentile(sim_prices, 50, axis=0)
            p75 = np.percentile(sim_prices, 75, axis=0)
            p90 = np.percentile(sim_prices, 90, axis=0)
            
            # Plot percentile bands
            plt.fill_between(range(self.simulation_length), p10, p90, alpha=0.15, color='blue', label="10-90% Range")
            plt.fill_between(range(self.simulation_length), p25, p75, alpha=0.25, color='blue', label="25-75% Range")
            plt.plot(p50, color='blue', linewidth=1.5, label="Median (50%)")
        
        # Plot individual paths
        indices = np.random.choice(self.num_simulations, num_to_plot, replace=False)
        for i, idx in enumerate(indices):
            plt.plot(sim_prices[idx], alpha=0.3, linewidth=0.8)
        
        # Overlay historical data if requested
        if show_historical and len(self.historical_data) > 0:
            hist_len = min(self.simulation_length, len(self.historical_data)) 
            
            # Get most relevant historical prices for comparison
            hist_prices = self.historical_data["close"].values[-hist_len:]
            
            # For forward simulation, we plot the historical data leading up to the starting point
            time_points = np.arange(-hist_len, 0)
            
            # Plot historical data with clear label
            plt.plot(time_points, hist_prices, color='black', linewidth=2, 
                    linestyle='--', label="Historical Close")
            
            # Update price range to include historical data
            min_price = min(min_price, np.min(hist_prices))
            max_price = max(max_price, np.max(hist_prices))
            
            # Add vertical line to mark transition from historical to simulated
            plt.axvline(x=0, color='red', linestyle='-', alpha=0.5, linewidth=1)
            plt.text(5, sim_prices[0, 0], "Simulation →", 
                    verticalalignment='center', alpha=0.7)
        
        # Set axis limits with some padding
        price_range = max_price - min_price
        plt.ylim(min_price - 0.05 * price_range, max_price + 0.05 * price_range)
        
        if show_historical:
            plt.xlim(-(hist_len+5), self.simulation_length+5)
        
        plt.title(f"Monte Carlo Simulated {price_type.capitalize()} Paths ({self.timeframe}, {self.simulation_length} periods)")
        plt.xlabel(f"Period ({self.timeframe} bars)")
        plt.ylabel(f"{price_type.capitalize()} Price (USD)")
        plt.grid(True, alpha=0.3)
        
        # Add legend, but only include items that are actually in the plot
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc="best")
        
        # Ensure y-axis reflects realistic price scale
        plt.ticklabel_format(style="plain", axis="y")  # Avoid scientific notation
        plt.tight_layout()
        plt.show()

    def calculate_indicators(self, indicator_list: list = None):
        """Calculate indicators on simulated paths (placeholder)."""
        if self.simulated_paths is None:
            raise ValueError("No simulated paths available. Run generate_price_paths first.")
        indicators = {}
        if indicator_list:
            for indicator in indicator_list:
                indicators[indicator] = None
        return indicators

    @abstractmethod
    def backtest_strategy(self):
        """Abstract method for backtesting a strategy on simulated paths."""
        pass