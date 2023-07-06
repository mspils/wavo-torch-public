"""
All utility functions that didn't fit anywhere else
"""
import pandas as pd

from typing import Union, Dict, Tuple
from dateutil.parser import ParserError


def is_float(s):
    """Tests if a string can be casted to float

    Args:
        s (_type_): _description_

    Returns:
        Boolean: True ist str is castable to float
    """
    if isinstance(s, int):
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def is_int(s):
    """Tests if a string can be casted to int

    Args:
        s (_type_): _description_

    Returns:
        Boolean: True ist str is castable to int
    """
    if isinstance(s, float) or (isinstance(s, str) and not s.isnumeric()):
        return False
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def get_objective_metric(s):
    """Return a reasonable 0th value for metrics, to not mess with chart scaling.
    Without a 0th value available at the start the hyperparameter tab doesn't work 

    Args:
        s (string): metric name

    Returns:
        float: 0th value for a metric
    """
    if s in ['nse', 'kge']:
        return 1
    else:
        return 0


class LargeGapError(Exception):
    """Basic Custom error, thrown if a gap in a dataset is too large

    Args:
        Exception (Exception):
    """


class StartIndexError(Exception):
    """Basic Custom error, thrown if a start value is given as a timestamp too close to the start of the input date

    Args:
        Exception (Exception):
    """


def get_start_index(start: Union[float, str],
                    df: pd.DataFrame,
                    in_size: int) -> int:
    """Finds the first index in a dataframe matching to either a percentage of the df or a timestamp.
        The index is the first value needed to make a prediction for (after?) the value start

    Args:
        start: float or timestamp from which to start the prediction.
        df : DataFrame in which the starting index is to be found
        in_size: input size, to adjust starting point if it is a timestamp

    Returns:
        starting index
    """
    if is_int(start):
        start = int(start)
        assert start >= 0
    elif is_float(start):
        start = float(start)
        assert start >= 0
        assert start <= 1
        start = int(start*len(df))
    else:
        try:
            start = pd.to_datetime(start)
        except (ValueError, TypeError, ParserError):
            print(
                f"Start value muss be a valid int, float or timestamp, but is {start}")

        start = df.index.get_indexer([start], method='nearest')[0]

        if start-in_size < 0:
            raise StartIndexError(
                f"Index {start} is too close to the beginning of the dataset. Must be a least {in_size}")
        start = start-in_size
    return start


def get_end_index(end: Union[float, str],
                  df: pd.DataFrame) -> int:
    """Finds the first index in a dataframe matching to either a percentage of the df or a timestamp.
    If this is a timestamp then the value will be the last row used for a prediction.
    An empty string mean using all data
    For ints and floats... in_size too few?

    Args:
        start: float or timestamp from which to start the prediction.
        df : DataFrame in which the starting index is to be found
        in_size: input size, to adjust starting point if it is a timestamp

    Returns:
        starting index
    """
    # TODO consistent handling between types (forgot what the problem is :( )
    if end is None or end == '':
        end = len(df)-1
    if is_int(end):
        end = int(end)
        assert end >= 0

    elif is_float(end):
        end = float(end)
        assert end >= 0
        assert end <= 1
        end = int(end*len(df))
    else:
        try:
            end = pd.to_datetime(end)
            # if pd.isnull(end):
            #    end = len(df)-1
        except (ValueError, TypeError, ParserError):
            print(
                f"Start value muss be a valid int, float or timestamp, but is {end}")

        end = df.index.get_indexer([end], method='nearest')[0]
        end = max(0, end+1)
    return end
