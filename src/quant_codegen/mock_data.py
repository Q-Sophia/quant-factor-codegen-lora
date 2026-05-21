from __future__ import annotations


def make_mock_ohlcv_frame(num_dates: int = 40, num_assets: int = 6):
    """Create a small OHLCV panel for executing generated factor functions."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=num_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(num_assets)]

    rows = []
    for symbol_index, symbol in enumerate(symbols):
        base = 20 + symbol_index * 3
        close = base + rng.normal(0, 0.8, size=num_dates).cumsum()
        close = np.maximum(close, 1.0)
        open_ = close * (1 + rng.normal(0, 0.01, size=num_dates))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.03, size=num_dates))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.03, size=num_dates))
        volume = rng.integers(100_000, 1_000_000, size=num_dates)
        amount = volume * close

        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": float(open_[i]),
                    "high": float(high[i]),
                    "low": float(low[i]),
                    "close": float(close[i]),
                    "volume": float(volume[i]),
                    "amount": float(amount[i]),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)
