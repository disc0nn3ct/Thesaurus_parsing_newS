"""Offline Raspberry Pi workload proxy: 100 series and 50 pairs; no providers."""
import argparse
import ctypes
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from thesaurus.market_analysis import correlations, forecast


def memory():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1024**2
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(months=None):
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-01-01', periods=252).strftime('%Y-%m-%d')
    data = [[{'date': d, 'value': float(v)} for d, v in zip(dates, 100 + np.cumsum(rng.normal(size=252)))] for _ in range(100)]
    metadata = {'frequency': 'daily', 'calendar': 'calendar', 'unit': '%', 'transform': 'difference', 'aggregation': 'last'}
    started = time.monotonic()
    tracemalloc.start()
    count = 0
    with threadpool_limits(limits=4):
        for rows in data:
            count += len(forecast(rows, metadata, created_at=dates[-1]+'T23:00:00+00:00',
                                  horizon={'value': months, 'unit': 'months'} if months else None))
        for i in range(50):
            correlations(data[2*i], data[2*i+1], metadata, metadata, window=120)
    peak = tracemalloc.get_traced_memory()[1]/1024**2
    tracemalloc.stop()
    result = {'series': 100, 'pairs': 50, 'forecasts': count, 'threads': 4,
              'horizon_months': months,
              'seconds': round(time.monotonic()-started, 3), 'python_peak_mb': round(peak, 2),
              'process_peak_mb': round(memory(), 2), 'platform': sys.platform,
              'note': 'Measured on current host; not a Raspberry Pi hardware measurement'}
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--months', type=int, choices=range(1, 25))
    run(parser.parse_args().months)
