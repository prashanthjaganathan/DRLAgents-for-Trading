"""OHLCV + Technical Indicators state representation."""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd


class OHLCVWithIndicators:
    """
    Extends raw OHLCV with technical indicators.

    State vector: [window_size * 5 OHLCV, RSI, MACD, MACD_signal, BB_upper, BB_lower, ATR, position_flag]

    Kept lean (6 indicators) per Risk II mitigation — avoids curse of dimensionality.
    """

    OHLCV_COLUMNS: typing.ClassVar[list[str]] = ["Open", "High", "Low", "Close", "Volume"]
    N_INDICATORS = 6  # RSI, MACD, MACD_signal, BB_upper, BB_lower, ATR

    def __init__(self, window_size: int = 20, rsi_period: int = 14) -> None:
        self.window_size = window_size
        self.rsi_period = rsi_period

    @property
    def obs_dim(self) -> int:
        """Total observation dimensionality."""
        ohlcv = self.window_size * len(self.OHLCV_COLUMNS)
        return ohlcv + self.N_INDICATORS + 1  # +1 for position flag

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicator columns to the DataFrame.

        Call this ONCE before training, not every step.
        The env should store the enriched DataFrame.
        """
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- RSI (14-period) ---
        df["RSI"] = self._compute_rsi(close, self.rsi_period)

        # --- MACD (12, 26, 9) ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # --- Bollinger Bands (20-period, 2 std) ---
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df["BB_upper"] = sma20 + 2 * std20
        df["BB_lower"] = sma20 - 2 * std20

        # --- ATR (14-period) ---
        df["ATR"] = self._compute_atr(high, low, close, period=14)

        # fill NaN from rolling windows with 0
        df.fillna(0, inplace=True)

        return df

    def build(self, df: pd.DataFrame, current_step: int, position: float) -> np.ndarray:
        """
        Build observation vector for the given step.

        Args:
            df:           DataFrame with OHLCV + precomputed indicator columns.
            current_step: Current index in the DataFrame.
            position:     1.0 if holding shares, 0.0 if flat.

        Returns:
            Flat float32 observation vector.
        """
        start = current_step - self.window_size

        # --- OHLCV window (normalized) ---
        window = df.iloc[start:current_step][self.OHLCV_COLUMNS].values
        col_min = window.min(axis=0)
        col_max = window.max(axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1.0
        window_norm = (window - col_min) / denom

        # --- Technical indicators at current step (normalized to stable ranges) ---
        row = df.iloc[current_step]
        current_close = row["Close"]

        indicators = np.array(
            [
                row["RSI"] / 100.0,  # RSI: 0-100 → 0-1
                row["MACD"] / current_close if current_close > 0 else 0.0,  # relative to price
                row["MACD_signal"] / current_close if current_close > 0 else 0.0,
                (row["BB_upper"] - current_close) / current_close if current_close > 0 else 0.0,
                (current_close - row["BB_lower"]) / current_close if current_close > 0 else 0.0,
                row["ATR"] / current_close if current_close > 0 else 0.0,  # relative volatility
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([window_norm.flatten(), indicators, [position]])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Indicator calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range — measures volatility."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period).mean()
