import numpy as np
import pandas as pd

from vrp.reports.cross_market import (
    assert_no_same_date_us_leakage,
    previous_us_trading_date_for_india,
)


def test_previous_us_trading_date_handles_mixed_datetime_units():
    india_dates = pd.Series(
        np.array(
            ["2024-01-02", "2024-01-03", "2024-01-08"],
            dtype="datetime64[ms]",
        )
    )
    us_dates = pd.Series(
        np.array(
            ["2024-01-02", "2024-01-05"],
            dtype="datetime64[us]",
        )
    )

    aligned = previous_us_trading_date_for_india(india_dates, us_dates)

    assert str(aligned["india_date"].dtype) == "datetime64[ns]"
    assert str(aligned["us_lagged_date"].dtype) == "datetime64[ns]"
    assert aligned.loc[0, "us_lagged_date"] is pd.NaT or pd.isna(
        aligned.loc[0, "us_lagged_date"]
    )
    assert aligned.loc[1, "us_lagged_date"] == pd.Timestamp("2024-01-02")
    assert aligned.loc[2, "us_lagged_date"] == pd.Timestamp("2024-01-05")

    assert_no_same_date_us_leakage(aligned)
