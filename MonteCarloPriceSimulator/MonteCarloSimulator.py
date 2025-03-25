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
        
        # Calculate mean reversion parameters
        price_series = np.log(self.historical_data["close"])
        price_diff = price_series - price_series.rolling(window=window).mean()
        # Drop NaN values and ensure sufficient data
        price_diff = price_diff.dropna()
        
        # If we have enough data, calculate mean reversion speed
        if len(price_diff) > 1:
            self.mean_reversion_speed = -np.polyfit(price_diff[:-1], np.diff(price_diff), 1)[0]
            # Ensure reasonable value (between 0 and 1 for stability)
            self.mean_reversion_speed = np.clip(self.mean_reversion_speed, 0, 1)
        else:
            # Default value if insufficient data
            self.mean_reversion_speed = 0.05

    def _fit_garch_model(self):
        """Fit GARCH(1,1) model to historical returns."""
        returns = np.log(self.historical_data["close"] / self.historical_data["close"].shift(1)).dropna()  # Remove scaling by 100
        model = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero", dist="Normal")
        result = model.fit(disp="off")
        self.model_params["garch"] = {
            "omega": result.params["omega"],
            "alpha": result.params["alpha[1]"],
            "beta": result.params["beta[1]"]
        }

    def _estimate_jump_params(self):
        """Estimate jump frequency and size from historical data."""
        returns = np.log(self.historical_data["close"] / self.historical_data["close"].shift(1)).dropna()
        # Define jumps as returns > 3σ (crude heuristic)
        jump_threshold = 3 * self.sigma
        jumps = returns[abs(returns) > jump_threshold]
        jump_freq = len(jumps) / len(returns)  # e.g., 0.02 = 2% chance per bar
        jump_std = jumps.std() if len(jumps) > 0 else 0.02  # Reduced default to 2% if no jumps
        
        self.model_params["jump"] = {
            "lambda": min(max(jump_freq, 0.001), 0.01),  # Between 0.1% and 1% chance
            "jump_mean": 0,
            "jump_std": min(jump_std, 0.03)  # Cap at 3%
        }

    def generate_price_paths(self, model_type: str = "GARCH+JUMP", return_type: str = "ohlc", 
                            model_params: dict = None):
        """Generate simulated price paths with enhanced regime switching."""
        params = model_params if model_params else self.model_params
        garch_params = params.get("garch", self.model_params["garch"])
        jump_params = params.get("jump", self.model_params["jump"])

        # Starting OHLC from first historical bar (fixed starting point)
        start_close = self.historical_data["close"].iloc[0]
        start_open = self.historical_data.get("open", start_close).iloc[0]
        start_high = self.historical_data.get("high", start_close).iloc[0]
        start_low = self.historical_data.get("low", start_close).iloc[0]

        if model_type == "GARCH+JUMP":
            closes = np.zeros((self.num_simulations, self.simulation_length))
            closes[:, 0] = start_close
            
            # Initialize regime states (randomly assign initial regimes)
            current_regimes = np.random.choice(
                ['bullish', 'bearish', 'high_vol', 'low_vol'], 
                size=self.num_simulations
            )
            
            # GARCH volatility simulation
            sigma = np.zeros((self.num_simulations, self.simulation_length))
            sigma[:, 0] = self.sigma
            omega, alpha, beta = garch_params["omega"], garch_params["alpha"], garch_params["beta"]
            
            # Jump components with regime-dependent parameters
            lambda_, jump_mean, jump_std = jump_params["lambda"], jump_params["jump_mean"], jump_params["jump_std"]
            jumps = np.random.poisson(lambda_, (self.num_simulations, self.simulation_length))
            jump_sizes = np.random.normal(jump_mean, jump_std, (self.num_simulations, self.simulation_length))
            
            # Simulate paths with regime switching
            for t in range(1, self.simulation_length):
                # Randomly switch regimes (5% chance per period)
                regime_switch = np.random.random(self.num_simulations) < 0.05
                if any(regime_switch):
                    current_regimes[regime_switch] = np.random.choice(
                        ['bullish', 'bearish', 'high_vol', 'low_vol'],
                        size=np.sum(regime_switch)
                    )
                
                # Update GARCH volatility
                prev_return = np.log(closes[:, t-1] / closes[:, t-2] if t > 1 else start_close / start_open)
                sigma[:, t] = np.sqrt(omega + alpha * prev_return**2 + beta * sigma[:, t-1]**2)
                
                # Generate returns with regime-dependent parameters
                z = np.random.normal(0, 1, self.num_simulations)
                
                # Apply regime-specific parameters
                mu = np.zeros(self.num_simulations)
                vol_multiplier = np.ones(self.num_simulations)
                
                for regime in self.regimes:
                    mask = current_regimes == regime
                    mu[mask] = self.regimes[regime]['mu']
                    vol_multiplier[mask] = self.regimes[regime]['sigma'] / self.sigma
                
                # Generate returns with GARCH + jumps + regime effects
                r = mu + sigma[:, t] * vol_multiplier * z + jump_sizes[:, t] * (jumps[:, t] > 0)
                
                # Add mean reversion component
                log_price = np.log(closes[:, t-1])
                mean_price = np.log(start_close)  # Could use rolling mean for more complexity
                mean_rev = self.mean_reversion_speed * (mean_price - log_price)
                r += mean_rev
                
                closes[:, t] = closes[:, t-1] * np.exp(r)

            # Generate OHLC if requested
            if return_type == "ohlc":
                paths = np.zeros((self.num_simulations, self.simulation_length, 4))
                paths[:, 0, 0] = start_open
                paths[:, 0, 1] = start_high
                paths[:, 0, 2] = start_low
                paths[:, 0, 3] = start_close
                
                for t in range(1, self.simulation_length):
                    paths[:, t, 3] = closes[:, t]
                    paths[:, t, 0] = closes[:, t-1]
                    r = np.log(closes[:, t] / closes[:, t-1])
                    k = 0.5
                    paths[:, t, 1] = closes[:, t] * np.exp(np.abs(r) + k * sigma[:, t])
                    paths[:, t, 2] = closes[:, t] * np.exp(-np.abs(r) - k * sigma[:, t])
                
                self.simulated_paths = paths
            else:
                self.simulated_paths = closes[:, :, np.newaxis]
            
        elif model_type == "GBM":
            closes = np.zeros((self.num_simulations, self.simulation_length))
            closes[:, 0] = start_close
            z = np.random.normal(0, 1, (self.num_simulations, self.simulation_length))
            for t in range(1, self.simulation_length):
                closes[:, t] = closes[:, t-1] * np.exp(self.mu + self.sigma * z[:, t])
            
            if return_type == "ohlc":
                paths = np.zeros((self.num_simulations, self.simulation_length, 4))
                paths[:, 0, 0] = start_open
                paths[:, 0, 1] = start_high
                paths[:, 0, 2] = start_low
                paths[:, 0, 3] = start_close
                for t in range(1, self.simulation_length):
                    paths[:, t, 3] = closes[:, t]
                    paths[:, t, 0] = closes[:, t-1]
                    r = np.log(closes[:, t] / closes[:, t-1])
                    paths[:, t, 1] = closes[:, t] * np.exp(np.abs(r) + 1.5 * self.sigma)
                    paths[:, t, 2] = closes[:, t] * np.exp(-np.abs(r) - 1.5 * self.sigma)
                self.simulated_paths = paths
            else:
                self.simulated_paths = closes[:, :, np.newaxis]
        
        return self.simulated_paths

    def plot_simulated_paths(self, num_paths_to_plot: int = 50, price_type: str = "close", 
                         show_historical: bool = True):
        """Plot a subset of simulated price paths with optional historical overlay.

        Args:
            num_paths_to_plot: Number of paths to plot (default: 10 for clarity).
            price_type: 'open', 'high', 'low', or 'close' (default: 'close').
            show_historical: Whether to overlay historical close prices (default: True).
        """
        if self.simulated_paths is None:
            raise ValueError("No simulated paths available. Run generate_price_paths first.")
        
        # Limit number of paths to plot
        num_to_plot = min(num_paths_to_plot, self.num_simulations)
        
        # Determine price index based on type
        price_idx = {"open": 0, "high": 1, "low": 2, "close": 3}.get(price_type, 3)
        
        plt.figure(figsize=(12, 6))
        
        # Plot simulated paths
        if self.simulated_paths.shape[2] == 1:  # close_only
            for i in range(num_to_plot):
                plt.plot(self.simulated_paths[i, :, 0], label=f"Sim Path {i+1}", alpha=0.5)
        else:  # ohlc
            for i in range(num_to_plot):
                plt.plot(self.simulated_paths[i, :, price_idx], label=f"Sim Path {i+1}", alpha=0.5)
        
        # Overlay historical data if requested
        if show_historical and price_type == "close":
            historical_prices = self.historical_data["close"].values[:self.simulation_length]
            plt.plot(historical_prices, label="Historical Close", color="black", linewidth=2, linestyle="--")
        
        plt.title(f"Monte Carlo Simulated {price_type.capitalize()} Paths ({self.timeframe}, {self.simulation_length} periods)")
        plt.xlabel("Period (15-minute bars)")
        plt.ylabel(f"{price_type.capitalize()} Price (USD)")
        plt.legend(loc="best")
        plt.grid(True)
        
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