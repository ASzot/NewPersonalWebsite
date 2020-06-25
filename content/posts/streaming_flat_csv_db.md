---
title: "Tea-time: Stream Financial Series Data from CSVs"
date: 2018-07-30
---

This post will describe a system I made called Tea-time. Tea-Time is for
reading financial time series data from a flat database of CSV files. 

  CSV files are a common format for financial time series data. For instance, you
  could store the pricing data for a stock in a file <code>stock_symb.csv</code>:

```
  Date,Open,High,Low,Close,Volume
  2018-07-02,24.320000,24.740000,24.320000,24.670000,61000
  2018-07-03,24.799999,25.010000,24.700001,24.830000,50900
  2018-07-05,24.850000,24.959999,24.690001,24.900000,59000
  2018-07-06,24.980000,25.030001,24.700001,24.790001,51400
  2018-07-09,24.950001,25.230000,24.830000,24.950001,92000
  2018-07-10,24.950001,25.100000,24.780001,24.830000,107200
  2018-07-11,24.750000,25.070000,24.750000,24.840000,81900
  2018-07-12,24.879999,24.980000,24.660000,24.969999,35000
  2018-07-13,24.969999,25.290001,24.920000,25.020000,55700
  2018-07-16,25.049999,25.129999,24.809999,24.920000,49000
  2018-07-17,24.920000,25.000000,24.809999,24.950001,58300
  2018-07-18,24.959999,25.000000,24.530001,24.780001,48800
  2018-07-19,24.780001,24.860001,24.590000,24.750000,49400
  2018-07-20,24.700001,24.990000,24.700001,24.730000,78100
  2018-07-23,24.730000,24.959999,24.730000,24.920000,49100
  2018-07-24,24.900000,24.950001,24.719999,24.809999,43200
  2018-07-25,24.799999,24.940001,24.379999,24.540001,44900
  2018-07-26,24.570000,24.730000,24.299999,24.629999,65700
  2018-07-27,24.719999,24.799999,24.299999,24.320000,50300
  2018-07-30,24.250000,24.600000,24.020000,24.090000,73277
  ...
```

This is the standard format for OHLCV (open, high, low, close, volume) data. It
is typical for data vendors to provide pricing data in this format. 

### How do we work with this data?
Pandas has some great functionality when it comes to working with CSV files.
Opening the CSV file, like the one above, would be as simple as:

```python
df = pd.read_csv('AAPL.csv', index_col=0, parse_dates=True)
```

This works fine for one symbol but what if you were working with many
symbols? Furthermore, what if each CSV files has <b>a lot</b> of rows? This could be
minute data or data across many years. The naive approach would be to load
the complete data for each symbol into memory.

```python
dfs = {}
for symb in test_symbs:
    df = pd.read_csv(CSV_DIR + '/' + test_symbs[0].symbol + '.csv', index_col = 0, parse_dates=True)
    dfs[symb.symbol] = df
results = pd.Panel.from_dict(results)
```

However, this consumes way too much memory needlessly. We don't need to have
the entire pricing history in memory. Typically, we only need a couple rows
for each symbol in memory. We can do a lot better in terms of memory and
performance than just storing plain CSV files in memory. 

### Tea-Time

  Tea-time is intended to be an optimized way of interacting with a directory
  of CSV files for time series data. Tea-time is built on the optimized <a
  href='http://discretelogics.com/teafiles/'>tea file</a> format.

  The first step of Tea-time is converting a regular directory full of CSV
  files to a directory full of compressed and optimized tea files. 

  Once that has been done we can easily work with the files. Teafiles are
  stored compressed and have an API for reading at arbitrary lines.  The
  system then can read rows at random access fairly quickly without the memory
  burden of storing the entire file. Furthermore, caching future results makes
  the system even faster. 

  Tea-time uses the assumption that the data will be read sequentially and could
  need a sliding window of dates. A certain number of rows
  are loaded ahead of time so that the system can provide the data even
  quicker. Tea-time also assumes that the data will be over multiple symbols on
  a defined calendar. Trading calendars are defined calendars that do not
  include every single day. The calendars Tea-time uses are from the <a
  href='https://pypi.org/project/trading-calendars/'>trading-calendar
  package</a>. This allows Tea-time to know which dates of the trading calendar
  to expect. Fetching data between mutliple symbols is synced up to the
  calendar. If data for a requested symbol is not found for that time
  NaN is returned. 

  I compared the performance of this system to Pandas in likely situations. The
  first test was how long it took to load in a symbol then multiple. The next
  test was reading rows that were cached. The test after that was reading a row
  right after the cached rows. Finally a row out of the cache and not
  sequential was read. The performance of the fetches is similar to Pandas but
  Tea-time can scale to indefinitely more symbols without much memory cost. 
  Below is the benchmark illustration.

{{< image src="/img/teatime_bench.png" class="center-image" width="400px" >}}

  The interface for fetching data by time and symbol is very simple using this
  system. Say you wanted to fetch the data for the last 50 days on date <code>2010-01-13</code>. That
  would be as simple as:

```python
  from trading_calendars import get_calendar

  cal = get_calendar('NYSE')
  dt = pd.to_datetime('2010-01-13')
  cacher = DataCacher(cal, './tea-files')
  results = cacher.get_symbs([symbol('AAPL'), symbol('MSFT')], dt, window=30)
```

  Note that the system does not use raw strings for symbols but uses an object
  that implements the <code>symbol</code> property giving the name of the
  ticker. This is so the system works well with <a
  href='https://github.com/quantopian/zipline'>Zipline</a>. 

  Overall, this sytem is optimized for reading rows sequentially for a large number
  of CSV files. It was built for reading stored financial data quickly. [View it
  on GitHub.](https://github.com/ASzot/tea-time)


