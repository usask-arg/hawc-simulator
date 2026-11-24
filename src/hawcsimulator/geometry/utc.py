from __future__ import annotations

import numpy as np
import pandas as pd


def utc_times(time) -> np.ndarray:
    """
    ``time`` as a one-dimensional array of timezone-naive UTC datetime64[ns].

    Parameters
    ----------
    time
        A time or sequence of times: anything pandas.to_datetime accepts. Times
        without a timezone are taken to be UTC.
    """
    times = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(time)))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    return times.to_numpy(dtype="datetime64[ns]")
