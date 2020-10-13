# import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
plt.close('all')

msft_df = pd.read_csv("MSFT.csv", parse_dates = ['date'])
date = msft_df["date"]
volume = msft_df["volume"]
high, low = msft_df["high"], msft_df["low"]
open, close = msft_df["open"], msft_df["close"]
range = abs(open - close)
msft_df['Range'] = range
days = 1000
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

def graph(time, *series, format="-"):  # time = last 'n' days
    for s in series:
        plt.plot(date[time:0:-1], s[time:0:-1], format, label=s.name)  # Reversing order because data is sorted by most recent
    plt.xlabel("Date")
    plt.ylabel("Price")

    ax = plt.axes()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, fontsize=8)


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True
plt.figure(num=1)
graph(days, high, low)
plt.legend(loc='best')
plt.figure(num=2)
graph(days, range)
plt.show()
