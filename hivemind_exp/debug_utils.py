import platform
import sys

import psutil

DIVIDER = "[---------] SYSTEM INFO [---------]"


def print_system_info():
    print(DIVIDER)
    print()
    print("Python Version:")
    print(f"  {sys.version}")

    print("\nPlatform Information:")
    print(f"  System: {platform.system()}")
    print(f"  Release: {platform.release()}")
    print(f"  Version: {platform.version()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")

    print("\nCPU Information:")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Total cores: {psutil.cpu_count(logical=True)}")
    cpu_freq = psutil.cpu_freq()
    print(f"  Max Frequency: {cpu_freq.max:.2f} Mhz")
    print(f"  Current Frequency: {cpu_freq.current:.2f} Mhz")

    print("\nMemory Information:")
    vm = psutil.virtual_memory()
    print(f"  Total: {vm.total / (1024**3):.2f} GB")
    print(f"  Available: {vm.available / (1024**3):.2f} GB")
    print(f"  Used: {vm.used / (1024**3):.2f} GB")

    print("\nDisk Information (>80%):")
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            disk_usage = psutil.disk_usage(partition.mountpoint)
            if disk_usage.used / disk_usage.total > 0.8:
                print(f"  Device: {partition.device}")
                print(f"    Mount point: {partition.mountpoint}")
                print(f"      Total size: {disk_usage.total / (1024**3):.2f} GB")
                print(f"      Used: {disk_usage.used / (1024**3):.2f} GB")
                print(f"      Free: {disk_usage.free / (1024**3):.2f} GB")
        except PermissionError:
            print("      Permission denied")

    print()
    print(DIVIDER)
