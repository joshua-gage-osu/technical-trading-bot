import pandas as pd
import numpy as np


# --- Helper Functions ---

# Logarithmic with zero handling
def zero_handled_log(series):
    return np.log(series.replace(0, np.nan))

#--- Moving Average Functions ---
#Simple Moving Average
def simple_moving_average(series, period):
    return series.rolling(period).mean()

#Weighted Moving Average
def weighted_moving_average(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

#Hull Moving Average
def hull_moving_average(series, period):
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half = weighted_moving_average(series, half_length)
    wma_full = weighted_moving_average(series, period)
    hull_ma = weighted_moving_average(2 * wma_half - wma_full, sqrt_length)
    return hull_ma

#Momentum Functions
#Momentum Using Simple MA for Volume and Price
def simple_momentum(volume_series, price_series, period):
    vol_ma = simple_moving_average(volume_series, period)
    price_ma = simple_moving_average(price_series, period)
    momentum = vol_ma * price_ma
    return momentum

#Momentum Using Weighted MA for Volume and Price
def weighted_momentum(volume_series, price_series, period):
    vol_ma = weighted_moving_average(volume_series, period)
    price_ma = weighted_moving_average(price_series, period)
    momentum = vol_ma * price_ma
    return momentum

#Momentum Using Hull MA for Volume and Price
def hull_momentum(volume_series, price_series, period):
    vol_ma = hull_moving_average(volume_series, period)
    price_ma = hull_moving_average(price_series, period)
    momentum = vol_ma * price_ma
    return momentum

#Simple Average Rocket Momentum Model
def simple_average_rocket_momentum(volume_series, price_series, period, thrust=1,spring_constant =.8):
    thurst_momentum = simple_momentum(volume_series, price_series, period)
    spring_anchor = simple_moving_average(price_series, period)
    spring_vector_signal = np.sign(price_series - spring_anchor)
    rocket_momentum = thrust * thurst_momentum - (spring_constant * spring_vector_signal* (price_series - spring_anchor))
    return rocket_momentum

    # Main Processing Function
def process_tqqq_data(output_path="processed_tqqq_data.csv"):

    # Read in the raw CSV file
    df = pd.read_csv("raw_tqqq_data.csv")

    # Make copy of original data from the CSV file
    df = df.copy()

    # Standardize column names with lowercase letters and no spaces
    df.columns = [c.strip().lower() for c in df.columns]

    # Sort by Date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Original Columns
    date = df['timestamp']
    open = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    adjclose = df['adjclose']
    volume = df['volume']

    # --- Logarithmic Price Transformations and their Derivatives ---

    #Price Data Logarithmic Transformations
    #Log Prices
    log_close = zero_handled_log(close)
    log_adjclose = zero_handled_log(adjclose)
    log_open = zero_handled_log(open)
    log_high = zero_handled_log(high)
    log_low = zero_handled_log(low)
    df['log_close'] = log_close
    df['log_adjclose'] = log_adjclose
    df['log_open'] = log_open
    df['log_high'] = log_high
    df['log_low'] = log_low

    #Logarithmic Adjclose Price Measure of Velocity and Acceleration
    df['log_velocity'] = log_adjclose.diff()
    df['log_acceleration'] = df['log_velocity'].diff()

    #Volume Logarithmic Transformations
    log_volume = zero_handled_log(volume)
    df['log_volume'] = log_volume


    # --- Raw Price Measure Moving Averages ---

    #20 Days
    # Adjusted Close Simple Moving Averages and their Derivatives
    day_sma_20 = simple_moving_average(adjclose, 20)
    df['20_day_sma'] = day_sma_20
    df['20_day_sma_velocity'] = day_sma_20.diff()
    df['20_day_sma_acceleration'] = df['20_day_sma_velocity'].diff()
    # Adjusted Close Weighted Moving Averages and their Derivatives
    day_wma_20 = weighted_moving_average(adjclose, 20)
    df['20_day_wma'] = day_wma_20
    df['20_day_wma_velocity'] = day_wma_20.diff()
    df['20_day_wma_acceleration'] = df['20_day_wma_velocity'].diff()
    # Adjusted Close Hull Moving Averages and their Derivatives
    day_hma_20 = hull_moving_average(adjclose, 20)
    df['20_day_hma'] = day_hma_20
    df['20_day_hma_velocity'] = day_hma_20.diff()
    df['20_day_hma_acceleration'] = df['20_day_hma_velocity'].diff()

    #50 Days
    # Adjusted Close Simple Moving Averages and their Derivatives
    day_sma_50 = simple_moving_average(adjclose, 50)
    df['50_day_sma'] = day_sma_50
    df['50_day_sma_velocity'] = day_sma_50.diff()
    df['50_day_sma_acceleration'] = df['50_day_sma_velocity'].diff()
    # Weighted Moving Averages and their Derivatives
    day_wma_50 = weighted_moving_average(adjclose, 50)
    df['50_day_wma'] = day_wma_50
    df['50_day_wma_velocity'] = day_wma_50.diff()
    df['50_day_wma_acceleration'] = df['50_day_wma_velocity'].diff()
    # Hull Moving Averages and their Derivatives
    day_hma_50 = hull_moving_average(adjclose, 50)
    df['50_day_hma'] = day_hma_50
    df['50_day_hma_velocity'] = day_hma_50.diff()
    df['50_day_hma_acceleration'] = df['50_day_hma_velocity'].diff()


    # --- Momentum Calculations ---
    #20 Day Momentums and their Derivatives
    #Simple Momentum
    df['20_day_simple_momentum'] = simple_momentum(volume, adjclose, 20)
    df['20_day_simple_momentum_delta'] = df['20_day_simple_momentum'].diff()
    #Weighted Momentum
    df['20_day_weighted_momentum'] = weighted_momentum(volume, adjclose, 20)
    df['20_day_weighted_momentum_delta'] = df['20_day_weighted_momentum'].diff()
    #Hull Momentum
    df['20_day_hull_momentum'] = hull_momentum(volume, adjclose, 20)
    df['20_day_hull_momentum_delta'] = df['20_day_hull_momentum'].diff()
    #Simple Average Rocket Momentum
    df['20_day_simple_average_rocket_momentum'] = simple_average_rocket_momentum(volume, adjclose, 20)
    df['20_day_simple_average_rocket_momentum_delta'] = df['20_day_simple_average_rocket_momentum'].diff()



    # --- Export Processed CSV file ---
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_tqqq_data()