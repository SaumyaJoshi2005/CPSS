from __future__ import annotations

import ctypes
import os
import subprocess
import sys


def memory_observation() -> dict:
    result = {"logical_cpus": os.cpu_count(), "process_rss_bytes": None, "process_peak_rss_bytes": None}
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
                ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
        state = MemoryStatus()
        state.length = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            result.update(total_ram_bytes=state.total, available_ram_bytes=state.available)
        class ProcessMemory(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in
                ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        counters = ProcessMemory()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if get_info(process(), ctypes.byref(counters), counters.cb):
            result.update(process_rss_bytes=counters.working, process_peak_rss_bytes=counters.peak)
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    return result


def hardware_observation() -> dict:
    result = memory_observation()
    try:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,driver_version", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10, **kwargs)
        result["gpu_inventory"] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_inventory"] = "unavailable"
    return result
