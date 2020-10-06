# import tensorflow as tf
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

msft_ds = pd.read_csv("MSFT.csv", parse_dates = ['date'])
date = msft_ds["date"]
volume = msft_ds["volume"]
high, low = msft_ds["high"], msft_ds["low"]
open, close = msft_ds["open"], msft_ds["close"]


def graph(time, *series, format="-"):  # time = last 'n' days
    for s in series:
        plt.plot(date[time:0:-1], s[time:0:-1], format, label=s.name)  # Reversing order because data is sorted by most recent
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)


plt.figure(figsize = (10, 6))
graph(1000, high, low, close)
leg = plt.legend(loc='best')
plt.show()
