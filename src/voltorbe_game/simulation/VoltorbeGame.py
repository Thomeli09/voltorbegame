# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:35:04 2025

@author: Thommes Eliott
"""

# Voltorb Game Solver Library

# Other Lib

# Custom Lib
from PlotLib import StartPlots, CloseAllPlots, PLTShow, PLTUpdateLayout, PLTScreenMaximize, ClosePlot

from VoltorbeGameLib import VoltorbGameGrid


if __name__ == "__main__":
    Game = VoltorbGameGrid(NCols=5, NRows=5)
    # XData
    Game.getXPoints =    [5, 7, 3, 5, 8]
    Game.getXVoltorbes = [2, 1, 3, 1, 0]
    # YData
    Game.getYPoints =    [5, 8, 5, 5, 5]
    Game.getYVoltorbes = [2, 1, 0, 2, 2]

    Game.VoltorbGameSolver()