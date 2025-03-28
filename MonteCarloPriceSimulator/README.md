MonteCarlo Price Simulator:(this is a base class)
-Generate Different price paths based on recent price action ( generate asymmetric volatility for more realistic market simulation)
-Handles OHLC data generation (not just close prices)
-Includes visualization capabilities ( left side of graph will have actual PA and right side will have simulated paths)
-Uses an abstract method pattern for strategy backtesting ( The child class must implement this to backtest trade strategy for all the generated price paths)

VolatilityBreakoutSimulator: ( child class):
-This is a simple backtesting implementation that
-Inherits from MonteCarloSimulator
-Implements a volatility breakout strategy
-Calculates rolling high/low ranges for breakout detection
-Handles risk management (profit target, stop loss, max holding period)
-Produces performance metrics (returns, win rate, drawdown, etc.)
-Includes equity curve generation

BacktraderVolatilityBreakoutSimulator: ( child class)
-This implementation uses the Backtrader framework and more sophisticated framework with order execution modeling
- this class  includes commission modeling ( probably shows less returns due to additional slippage calculaions)
-inherits from MonteCarloSimulator
-Implements the same volatility breakout strategy
-Uses Backtrader's architecture (Strategy class, Cerebro engine)
-Handles the same risk management parameters
-Generates similar performance metrics