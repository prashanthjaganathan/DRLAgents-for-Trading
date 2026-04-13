"""Raw OHLCV state representation with min-max normalization."""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd


class RawOHLCV:
    """
    Extracts a sliding window of raw OHLCV data, normalized per-column.

    State vector: [window_size * 5 OHLCV values, position_flag]
    """

    COLUMNS: typing.ClassVar[list[str]] = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size

    @property
    def obs_dim(self) -> int:
        """Total observation dimensionality."""
        return self.window_size * len(self.COLUMNS) + 1  # +1 for position flag

    def build(self, df: pd.DataFrame, current_step: int, position: float) -> np.ndarray:
        """
        Build observation vector for the given step.

        Args:
            df:           Full OHLCV DataFrame.
            current_step: Current index in the DataFrame.
            position:     1.0 if holding shares, 0.0 if flat.

        Returns:
            Flat float32 observation vector.
        """
        start = current_step - self.window_size
        window = df.iloc[start:current_step][self.COLUMNS].values

        # min-max normalize each column within the window
        col_min = window.min(axis=0)
        col_max = window.max(axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1.0
        window_norm = (window - col_min) / denom

        obs = np.concatenate([window_norm.flatten(), [position]])
        return obs.astype(np.float32)
