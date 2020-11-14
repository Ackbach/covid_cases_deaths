import pandas as pd
import os.path as path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import datetime as dt

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

data_folder = '~/Projects/covid_cases_deaths/DATA'
filepath = path.join(data_folder, 'all-states-history.csv')

df = pd.read_csv(filepath)
df = df[['date', 'state', 'death', 'positive']]


def lag_finder(y1, y2, sr, state: str) -> list:
    n = len(y1)

    # What must y2 be shifted by to match y1?
    corr = signal.correlate(y1, y2, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)]
                                                           * signal.correlate(y2, y2, mode='same')[int(n/2)])

    # Set up the delay array.
    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr - 1, n)
    delay = delay_arr[np.argmax(corr)]
    print('For state ' + state + ', deaths must be shifted by ' + str(delay) +
          ' days to match cases; max correlation is ' + str(np.max(corr)))

    return [np.max(corr), delay_arr[np.argmax(corr)]]


def debug_print(my_df: pd.DataFrame, msg: str) -> None:
    print()
    print(msg)
    print()
    print(my_df.info())
    print(my_df.describe())
    print(my_df.head(20))
    print(my_df.tail(20))
    print()


debug_print(df, 'After reading. ')
# Plan:
# 1. Impute missing values with 0.
# 2. Replace weekend values with linearly interpolated values.
# 3. Run convolution of death with positive, on a per-state basis.
# 4. Find which date difference maxes out the correlation.
# 5. On a per-state basis, report the resulting correlations.

# 1.
df = df.replace(np.nan, 0)
debug_print(df, 'After replacing NaN with 0.')
df['date'] = df['date'].apply(pd.to_datetime)
debug_print(df, 'After converting to datetime.')
df['is_weekday'] = df['date'].apply(
    lambda x: dt.datetime.weekday(x) < 5)
debug_print(df, 'After determining weekdays.')
states = set(df['state'])


def analyze_state(state: str) -> list:
    """
    We perform the analysis on a state, and report the results at the end.
    The return list contains first the
    :param state:
    :return:
    """

    # Work with MN only, for now.
    state_df = df.loc[state == df['state']]
    state_df['cases dy'] = np.zeros(len(state_df))
    state_df['deaths dy'] = np.zeros(len(state_df))

    # Smooth the cumulative data over weekends.
    i = 0
    while i < len(state_df):

        # Find beginning of a weekend.
        while state_df['is_weekday'].iloc[i] and i < len(state_df) - 1:
            i += 1

        # i now corresponds to Sunday, as we go backwards through time.
        monday_i = i - 1
        sunday_i = i
        saturday_i = i + 1
        friday_i = i + 2

        if friday_i < len(state_df):
            state_df['positive'].iloc[saturday_i] = (2 / 3.) * state_df['positive'].iloc[friday_i] \
                + (1 / 3.) * state_df['positive'].iloc[monday_i]
            state_df['positive'].iloc[sunday_i] = (1 / 3.) * state_df['positive'].iloc[friday_i] \
                + (2 / 3.) * state_df['positive'].iloc[monday_i]
            state_df['death'].iloc[saturday_i] = (2 / 3.) * state_df['death'].iloc[friday_i] \
                + (1 / 3.) * state_df['death'].iloc[monday_i]
            state_df['death'].iloc[sunday_i] = (1 / 3.) * state_df['death'].iloc[friday_i] \
                + (2 / 3.) * state_df['death'].iloc[monday_i]

        i = friday_i

    # Calculate discrete differential.
    for i in range(len(state_df) - 1):
        state_df['cases dy'].iloc[i] = state_df['positive'].iloc[i] - state_df['positive'].iloc[i+1]
        state_df['deaths dy'].iloc[i] = state_df['death'].iloc[i] - state_df['death'].iloc[i+1]

    # Now we run the correlations and pick the largest.
    max_corr, delay = lag_finder(state_df['cases dy'], state_df['deaths dy'], 1, state)

    return [state, max_corr, delay]


states_df = pd.DataFrame(columns=['state', 'max_corr', 'delay'])
i = 0
analyze_state('CA')

debug_print(states_df, 'All states')
