# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:13:23 2025

@author: Thommes Eliott
"""

# Time management library


# Other Lib
import time
import matplotlib.pyplot as plt
from typing import Tuple


# Custom Lib
from PlotLib import ParamPLT, StartPlots, PLTShow, PLTBar


"""
Chronometer structure
"""


class Chrono:
    def __init__(self, Name):
        """
        Initializes the stopwatch with a name.
        """
        self.Name = Name
        self.StartTime = None
        self.ElapsedTime = 0
        self.Running = False
        self.Records = []

    @property
    def getName(self):
        return self.Name

    @getName.setter
    def getName(self, Name):
        self.Name = Name

    @property
    def getStartTime(self):
        return self.StartTime

    @getStartTime.setter
    def getStartTime(self, Time):
        self.StartTime = Time

    @property
    def getElapsedTime(self):
        return self.ElapsedTime

    @getElapsedTime.setter
    def getElapsedTime(self, ElapsedTime):
        self.ElapsedTime = ElapsedTime

    @property
    def getRunning(self):
        return self.Running

    @getRunning.setter
    def getRunning(self, Running):
        self.Running = Running

    @property
    def getRecords(self):
        return self.Records

    @getRecords.setter
    def getRecords(self, Record):
        self.Records.append(Record)

    @property
    def Start(self):
        """
        Starts or resumes the stopwatch.
        """
        if (not self.getRunning):
            self.getStartTime = time.time()
            self.getRunning = True
            print("Info: Stopwatch started.")
        else:
            print("Info: Stopwatch already running.")

    @property
    def Stop(self):
        """
        Stops the stopwatch and records the elapsed time.
        """
        if self.getRunning:
            self.getElapsedTime += time.time() - self.getStartTime
            self.getRunning = False
            print("Info: Stopwatch stopped.")
        else:
            print("Info: Stopwatch already stopped.")

    @property
    def Reset(self):
        """
        Resets the stopwatch.
        """
        self.getStartTime = None
        self.getElapsedTime = 0
        self.getRunning = False
        print("Info: Stopwatch reset.")

    def Record(self, label=None):
        """
        Records the current elapsed time with an optional label.
        """
        if self.getRunning:
            self.Stop
        self.getRecords = (self.ElapsedTime, label)
        self.getElapsedTime = 0  # Resets elapsed time after recording
        print("Info: Stopwatch time recorded.")

    def PLTRecords(self, paramPLT=None):
        """
        Displays the records as a graph.
        """
        if not self.getRecords:
            print("Info: No records to display.")
            return

        if paramPLT is None:
            paramPLT = ParamPLT(colour='b', linetype=1, marker=0, linesize=0.8, fontsize=14)
            paramPLT.getTitle = f"Stopwatch: {self.getName}"
            paramPLT.getXLabel = "Records"
            paramPLT.getYLabel = "Elapsed time (s)"
            paramPLT.getGrid(Axis=-5)

        Times = [rec[0] for rec in self.getRecords]
        Labels = [rec[1] if rec[1] else f"Record {i+1}" for i, rec in enumerate(self.getRecords)]

        StartPlots()
        PLTBar(Labels, Times, paramPLT)
        PLTShow(paramPLT)


# Utility function to convert time to optimal unit
def Time2OptiTime(Value: float, Unit: str, NDecimals: int = 3, ReturnAsStr: bool = False) -> Tuple[float, str] | str:
    """
    Convert a time quantity to the most readable unit.

    - Exact conversions (no heuristics beyond "largest Unit with Value >= 1").
    - Handles sub-second (ns, µs, ms) up to years.
    - Negative values supported.
    """

    # Accept common aliases → seconds multiplier
    UnitStr = Unit.strip().lower()
    Alias = {
        # base + aliases
        "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
        # sub-second
        "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
        "us": 1e-6, "µs": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
        "ns": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
        # supra-second
        "min": 60.0, "m": 60.0, "minute": 60.0, "minutes": 60.0,
        "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
        "d": 86400.0, "day": 86400.0, "days": 86400.0,
        "wk": 604800.0, "w": 604800.0, "week": 604800.0, "weeks": 604800.0,
        "mo": 2592000.0, "month": 2592000.0, "months": 2592000.0,  # 1 month = 30 days
        "yr": 31557600.0, "y": 31557600.0, "year": 31557600.0, "years": 31557600.0,  # 1 year = 365.25 days
    }
    if UnitStr not in Alias:
        raise ValueError(f"Unknown unit '{Unit}'")

    # Canonical display order (small → large)
    Units = [
        ("ns", 1e-9),
        ("µs", 1e-6),
        ("ms", 1e-3),
        ("s", 1.0),
        ("min", 60.0),
        ("h", 3600.0),
        ("d", 86400.0),
        ("wk", 604800.0),
        ("mo", 2592000.0),  # 1 month = 30 days
        ("yr", 31557600.0),  # 1 year = 365.25 days
    ]

    Seconds = float(Value) * Alias[UnitStr]
    if Seconds == 0:
        Out = (0.0, "s")
        return Out if not ReturnAsStr else "0 s"

    Sign = -1.0 if Seconds < 0 else 1.0
    AbsSeconds = abs(Seconds)

    # Find the largest unit whose factor ≤ |Seconds|
    Index = 0
    for i, (_, Factor) in enumerate(Units):
        if Factor <= AbsSeconds:
            Index = i
        else:
            break

    Label, Factor = Units[Index]
    Val = round(Sign * (AbsSeconds / Factor), NDecimals)

    return (Val, Label) if not ReturnAsStr else f"{Val:.{NDecimals}f} {Label}"

"""
Usage example
"""
if __name__ == "__main__":
    # Chrono example
    DeltaTime = Chrono("My Stopwatch")
    DeltaTime.Start
    time.sleep(2)  # Simulates a computation
    DeltaTime.Record("Task 1")
    DeltaTime.Reset

    DeltaTime.Start
    time.sleep(1)  # Simulates another computation
    DeltaTime.Record("Task 2")

    DeltaTime.PLTRecords()

    # Time conversion examples
    print(Time2OptiTime(0.00042, "s"))      # (420.0, 'µs')
    print(Time2OptiTime(0.42, "s"))         # (420.0, 'ms')
    print(Time2OptiTime(59, "s"))           # (59.0, 's')
    print(Time2OptiTime(90, "s"))           # (1.5, 'min')
    print(Time2OptiTime(5400, "s"))         # (1.5, 'h')
    print(Time2OptiTime(400000, "ms"))      # (6.667, 'min')
    print(Time2OptiTime(-2500, "ms"))       # (-2.5, 's')
    print(Time2OptiTime(1.2e-10, "s"))      # (0.12, 'ns')  # shows <1 in the smallest unit
    print(Time2OptiTime(365.25*24*60*60*10, "s", ReturnAsStr=True))  # "1.111 h"
