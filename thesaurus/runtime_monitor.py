"""Independent service heartbeat and bounded resource samples, using stdlib only."""
import asyncio
import contextlib
import ctypes
import logging
import os
import platform
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .monitoring import safe_record
from .store import dumps, utcnow

logger = logging.getLogger(__name__)


def resources(path):
    result = {'cpu_cores': os.cpu_count(), 'process_rss_mb': None,
              'system_memory_available_mb': None, 'system_memory_total_mb': None}
    if os.name == 'nt':
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ('ullTotalPhys','ullAvailPhys','ullTotalPageFile',
                    'ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
        if kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            result.update(system_memory_available_mb=status.ullAvailPhys/1048576,
                          system_memory_total_mb=status.ullTotalPhys/1048576)
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            result['process_rss_mb'] = counters.WorkingSetSize/1048576
    elif Path('/proc/self/statm').exists():
        pages = int(Path('/proc/self/statm').read_text().split()[1])
        result['process_rss_mb'] = pages*os.sysconf('SC_PAGE_SIZE')/1048576
        values = {line.split(':')[0]: int(line.split()[1])/1024 for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith(('MemTotal:', 'MemAvailable:'))}
        result['system_memory_available_mb'] = values.get('MemAvailable')
        result['system_memory_total_mb'] = values.get('MemTotal')
    disk = shutil.disk_usage(Path(path).resolve().parent)
    result.update(disk_free_gb=disk.free/1073741824, disk_total_gb=disk.total/1073741824)
    result['database_mb'] = sum(p.stat().st_size for p in [Path(path),Path(str(path)+'-wal'),Path(str(path)+'-shm')]
                                if p.is_file())/1048576
    return result


def telegram_connected(client):
    connected = getattr(client, 'is_connected', False) is True
    session = getattr(client, 'session', None)
    signal = getattr(session, 'is_connected', None)
    if signal is not None and callable(getattr(signal, 'is_set', None)):
        return connected and signal.is_set() is True
    return connected


class RuntimeMonitor:
    def __init__(self, store, client):
        self.store, self.client = store, client
        self.identity = uuid4().hex
        self._task = None
        self._last_sample = None
        self._last_cleanup = None
        self._last_cpu = time.process_time()
        self._last_clock = time.monotonic()
        self.heartbeat_interval = 10
        self._pulse_lock = Lock()
        self._finished = False

    def start(self):
        # The CLI calls this only while holding the single-worker process lock.
        with self.store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            previous = conn.execute("SELECT id FROM monitor_runtime WHERE state IN ('starting','running')").fetchall()
            for row in previous:
                # The last heartbeat is the last known observation, not a fabricated shutdown time.
                conn.execute("UPDATE monitor_runtime SET state='interrupted',telegram_connected=0 WHERE id=?", (row['id'],))
            timestamp = utcnow()
            conn.execute('''INSERT INTO monitor_runtime(id,pid,host,started_at,heartbeat_at,state)
                VALUES(?,?,?,?,?,'starting')''', (self.identity,os.getpid(),platform.node(),timestamp,timestamp))
        for row in previous:
            safe_record(self.store, 'service', 'previous_run_interrupted', outcome='interrupted', level='warning',
                        entity_type='monitor_runtime', entity_id=row['id'])
        safe_record(self.store, 'service', 'started', entity_type='monitor_runtime', entity_id=self.identity)
        self.pulse()

    def pulse(self):
        with self._pulse_lock:
            if not self._finished:
                self._pulse()

    def _pulse(self):
        clock, cpu = time.monotonic(), time.process_time()
        connected = telegram_connected(self.client)
        prior = self.store.one('SELECT state,telegram_connected FROM monitor_runtime WHERE id=?', (self.identity,))
        state = 'running' if connected or prior and prior['state']=='running' else 'starting'
        self.store.execute("UPDATE monitor_runtime SET heartbeat_at=?,telegram_connected=?,state=? WHERE id=?",
                           (utcnow(),int(connected),state,self.identity))
        if prior and bool(prior['telegram_connected']) != connected:
            safe_record(self.store, 'service', 'telegram_connected' if connected else 'telegram_disconnected',
                        outcome='ok' if connected else 'disconnected', level='info' if connected else 'warning',
                        entity_type='monitor_runtime', entity_id=self.identity)
        if self._last_sample is None or clock-self._last_sample >= 60:
            values = resources(self.store.path)
            elapsed = clock-self._last_clock
            values['process_cpu_percent'] = max(0,(cpu-self._last_cpu)/elapsed*100) if self._last_sample is not None and elapsed>0 else None
            self.store.execute('INSERT INTO monitor_samples(runtime_id,created_at,metrics) VALUES(?,?,?)',
                               (self.identity,utcnow(),dumps(values)))
            self._last_sample, self._last_clock, self._last_cpu = clock, clock, cpu
            if self._last_cleanup is None or clock-self._last_cleanup >= 3600:
                cutoff = (datetime.now(timezone.utc)-timedelta(days=90)).isoformat()
                self.store.execute('DELETE FROM monitor_samples WHERE created_at<?', (cutoff,))
                self._last_cleanup = clock

    def finish(self, error=None):
        with self._pulse_lock:
            self._finished = True
            self._finish(error)

    def _finish(self, error=None):
        self.store.execute('UPDATE monitor_runtime SET state=?,finished_at=?,telegram_connected=0 WHERE id=?',
                           ('failed' if error else 'stopped',utcnow(),self.identity))
        safe_record(self.store, 'service', 'failed' if error else 'stopped', outcome='failed' if error else 'ok',
                    level='error' if error else 'info', entity_type='monitor_runtime', entity_id=self.identity,
                    details={'error_type':type(error).__name__} if error else {})

    async def _tick(self):
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                sample = asyncio.create_task(asyncio.to_thread(self.pulse))
                try:
                    await asyncio.shield(sample)
                except asyncio.CancelledError:
                    # The thread must finish before finish() records the stopped state.
                    await sample
                    raise
            except Exception as exc:  # noqa: BLE001 - Resource sampling must not stop the service.
                logger.warning('Service heartbeat failed (%s)', type(exc).__name__)

    async def __aenter__(self):
        await asyncio.to_thread(self.start)
        self._task = asyncio.create_task(self._tick())
        return self

    async def __aexit__(self, kind, error, traceback):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await asyncio.to_thread(self.finish, error)
