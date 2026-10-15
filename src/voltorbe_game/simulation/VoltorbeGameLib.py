# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:35:04 2025

@author: Thommes Eliott
"""

# Voltorb Game Solver Library
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Custom Lib
from TimeLib import Chrono
from PlotLib import ParamPLT, StartPlots, CloseAllPlots, PLTShow, DefaultParamPLT
from PlotLib import PLTUpdateLayout, PLTScreenMaximize, PLTImShow, PLTMultiPlot

class VoltorbGameGrid:
    def __init__(self, NCols=None, NRows=None):
        # Game parameters
        self.NRows = NRows
        self.NCols = NCols
        self.MaxVal = 3 # All values are between 0 and MaxVal included

        # Game initial data
        self.ArrayXPoints = None
        self.ArrayXVoltorbes = None
        self.ArrayYPoints = None
        self.ArrayYVoltorbes = None
        
        # Game optional data
        self.MatrixDataTiles = None # Matrix to hold the data of the tiles
        # Values below 0 are hidden tiles, values above 0 are the values of the revealed tiles

        # Computed data
        self.LPossibleGrid = []  # List to hold possible grids

        self.MatrixProb = None # Matrix to hold the probability of each tile being a 0

        self.MatrixCmptScores = None # Matrix to hold the computed scores to choose the best tile to reveal

    # Game parameters
    @property
    def getNRows(self):
        return self.NRows

    @getNRows.setter
    def getNRows(self, NRows):
        self.NRows = NRows

    @property
    def getNCols(self):
        return self.NCols

    @getNCols.setter
    def getNCols(self, NCols):
        self.NCols = NCols

    @property
    def getMaxVal(self):
        return self.MaxVal

    @getMaxVal.setter
    def getMaxVal(self, MaxVal):
        self.MaxVal = MaxVal

    # Game initial data
    @property
    def getXPoints(self):
        return self.ArrayXPoints

    @getXPoints.setter
    def getXPoints(self, ArrayVal):
        # Convert the list to a NumPy array if needed
        if isinstance(ArrayVal, list):
            ArrayVal = np.array(ArrayVal)
        # Check if the size of the array is correct
        if ArrayVal.size != self.getNCols:
            print("Error: ArrayXPoints must have the same size as the number of columns.")
            return False
        self.ArrayXPoints = ArrayVal

    @property
    def getXVoltorbes(self):
        return self.ArrayXVoltorbes

    @getXVoltorbes.setter
    def getXVoltorbes(self, ArrayVal):
        # Convert the list to a NumPy array if needed
        if isinstance(ArrayVal, list):
            ArrayVal = np.array(ArrayVal)
        # Check if the size of the array is correct
        if ArrayVal.size != self.getNCols:
            print("Error: ArrayXVoltorbes must have the same size as the number of columns.")
            return False
        self.ArrayXVoltorbes = ArrayVal

    @property
    def getYPoints(self):
        return self.ArrayYPoints

    @getYPoints.setter
    def getYPoints(self, ArrayVal):
        # Convert the list to a NumPy array if needed
        if isinstance(ArrayVal, list):
            ArrayVal = np.array(ArrayVal)
        # Check if the size of the array is correct
        if ArrayVal.size != self.getNRows:
            print("Error: ArrayYPoints must have the same size as the number of rows.")
            return
        self.ArrayYPoints = ArrayVal

    @property
    def getYVoltorbes(self):
        return self.ArrayYVoltorbes

    @getYVoltorbes.setter
    def getYVoltorbes(self, ArrayVal):
        # Convert the list to a NumPy array if needed
        if isinstance(ArrayVal, list):
            ArrayVal = np.array(ArrayVal)
        # Check if the size of the array is correct
        if ArrayVal.size != self.getNRows:
            print("Error: ArrayYVoltorbes must have the same size as the number of rows.")
        self.ArrayYVoltorbes = ArrayVal
    
    # Game optional data
    @property
    def getMatrixDataTiles(self):
        if self.MatrixDataTiles is None:
            # Initialize the matrix with -1 (hidden tiles)
            self.MatrixDataTiles = np.zeros((self.getNRows, self.getNCols)) - 1
        return self.MatrixDataTiles

    @getMatrixDataTiles.setter
    def getMatrixDataTiles(self, MatrixDataTiles):
        self.MatrixDataTiles = MatrixDataTiles

    def AddTile2MatrixDataTiles(self, Val, Row, Column):
        Matrix = self.getMatrixDataTiles
        Matrix[Row, Column] = Val
        self.getMatrixDataTiles = Matrix

    # Computed data
    @property
    def getLPossibleGrid(self):
        return self.LPossibleGrid

    @getLPossibleGrid.setter
    def getLPossibleGrid(self, LPossibleGrid):
        self.LPossibleGrid = LPossibleGrid

    @property
    def getMatrixProb(self):
        if self.MatrixProb is None:
            # Initialize the matrix with 1 (100% probability of being a 0) because no data is known
            self.MatrixProb = np.zeros((self.getNRows, self.getNCols)) + 1
        return self.MatrixProb

    @getMatrixProb.setter
    def getMatrixProb(self, MatrixProb):
        self.MatrixProb = MatrixProb

    @property
    def getMatrixCmptScores(self):
        if self.MatrixCmptScores is None:
            # Initialize the matrix with 0
            self.MatrixCmptScores = np.zeros((self.getNRows, self.getNCols))
        return self.MatrixCmptScores

    @getMatrixCmptScores.setter
    def getMatrixCmptScores(self, MatrixCmptScores):
        self.MatrixCmptScores = MatrixCmptScores

    # Methods

    # Data aquisition methods
    def VolorbeInputGene(self):
        """
        Ask for the row and column of the tiles and the value of the tiles
        """
        # Ask for the row and column of the tiles
        TempStatus = True
        while TempStatus:
            print("Querying the row of the tile")
            try:
                Row = int(input())
            except ValueError:
                print("Error: Please enter a Int value.")

            print("Querying the column of the tile")
            try:
                Col = int(input())
            except ValueError:
                print("Error: Please enter a Int value.")

            # As long as the position is invalid, the player has to give a valid position
            if Row < 0 or Row >= self.getNRows or Col < 0 or Col >= self.getNCols:
                print("Error: Invalid position.")
            else:
                TempStatus = True

        # Ask for the value at the selected position
        Value = self.VolorbeInputSpec(Row, Col)
        return Row, Col, Value

    def VolorbeInputSpec(self, Row, Col):
        """
        Ask for the value of tiles in row row and column col
        """
        # Ask for the value at the selected position
        print(f"Querying value at position ({Row}, {Col})")
        TempStatus = True
        while TempStatus:
            try:
                Value = int(input())
                TempStatus = False
            except ValueError:
                print("Error: Please enter a Int value.")
    
        return Value


    # Computations methods
    def CmptScores(self):
        """
        Compute the scores for each position based on the point and voltorbe vectors and the current knowledge.
        """
        Rows, Cols = self.getNRows, self.getNCols
        MaxValue = self.getMaxVal
        MatrixDataTiles = self.getMatrixDataTiles
        XPoints = self.getXPoints[:Cols].flatten()  # Take the first `Cols` elements
        YPoints = self.getYPoints[:Rows].flatten()  # Take the first `Rows` elements
        XVoltorbes = self.getXVoltorbes[:Cols].flatten()  # Take the first `Cols` elements
        YVoltorbes = self.getYVoltorbes[:Rows].flatten()  # Take the first `Rows` elements

        # Verify that the sum of the points and voltorbes in X and Y vectors match
        if np.sum(YPoints) != np.sum(XPoints) or np.sum(YVoltorbes) != np.sum(XVoltorbes):
            print("Error: Different total number of points or voltorbes on the whole matrix.")
            return

        # Compute the number of unknown tiles for each row and column and put it inside a np array
        RowUnknownTiles = np.array([np.sum(MatrixDataTiles[i] < 0) for i in range(Rows)])
        ColUnknownTiles = np.array([np.sum(MatrixDataTiles[:, j] < 0) for j in range(Cols)])

        # Compute the scores for each position
        # Formula is:  The number of points in the row decreased by the number of already known points in the row all divided by the number of unknown tiles in the row minus the number unknown voltorbes in the row (Number of voltorbes in the row minus the number of already known voltorbes in the row)
        # the same formula for the column to get the column score and take the minimum of the two scores to get the score of the position
        ScoresMatrix = np.zeros((Rows, Cols))
        for i in range(Rows):
            RowScoreDenom = RowUnknownTiles[i] - (YVoltorbes[i] - np.sum(MatrixDataTiles[i] == 0))
            if RowScoreDenom == 0:
                RowScore = 0
            elif (YVoltorbes[i] - np.sum(MatrixDataTiles[i] == 0)) <= 0:
                    RowScore = np.inf
            else:
                RowScore = (YPoints[i] - np.sum((MatrixDataTiles[i] > 0)*MatrixDataTiles[i])) / (RowUnknownTiles[i] - (YVoltorbes[i] - np.sum(MatrixDataTiles[i] == 0)))

            for j in range(Cols):
                ColScoreDenom = ColUnknownTiles[j] - (XVoltorbes[j] - np.sum(MatrixDataTiles[:, j] == 0))
                if ColScoreDenom == 0:
                    ColScore = 0
                elif (XVoltorbes[j] - np.sum(MatrixDataTiles[:, j] == 0)) <= 0:
                    ColScore = np.inf
                else:
                    ColScore = (XPoints[j] - np.sum((MatrixDataTiles[:, j] > 0)*MatrixDataTiles[:, j])) / (ColUnknownTiles[j] - (XVoltorbes[j] - np.sum(MatrixDataTiles[:, j] == 0)))

                if MatrixDataTiles[i, j] < 0:
                    ScoresMatrix[i, j] = min(RowScore, ColScore)
                    if RowScore <= 1 or ColScore <= 1:
                        ScoresMatrix[i, j] = 0  # If the score is less or equal to 1, the tile is a voltorbe or is a one point tile, which is not interesting
                    if RowScore == np.inf or ColScore == np.inf:
                        ScoresMatrix[i, j] = np.inf
                else:
                    ScoresMatrix[i, j] = MatrixDataTiles[i, j]*0  # Known values are kept as to not be selected

        # Update the computed scores matrix
        self.getMatrixCmptScores = ScoresMatrix
        
    def CmptRowCombinations(self):
        """
        Compute all valid row combinations based on the point and voltorbe vectors and the current knowledge.

        # Improvements:
        - Use the list of possible grids to compute the valid row combinations if existing
        """
        Rows, Cols = self.getNRows, self.getNCols
        MaxValue = self.getMaxVal
        MatrixDataTiles = self.getMatrixDataTiles  # Known tiles are used to filter valid rows (-1 are unknown tiles))

        # Dynamically slice the Points and Voltorbes vectors to match the current grid size
        XPointsSubset = self.getXPoints[:Cols].flatten()  # Take the first `Cols` elements
        YPointsSubset = self.getYPoints[:Rows].flatten()  # Take the first `Rows` elements
        XVoltorbesSubset = self.getXVoltorbes[:Cols].flatten()  # Take the first `Cols` elements
        YVoltorbesSubset = self.getYVoltorbes[:Rows].flatten()  # Take the first `Rows` elements

        # Verify that the sum of the points and voltorbes in X and Y vectors match
        if np.sum(YPointsSubset) != np.sum(XPointsSubset) or np.sum(YVoltorbesSubset) != np.sum(XVoltorbesSubset):
            print("Error: Different total number of points or voltorbes on the whole matrix.")
            self.getLPossibleGrid = []  # Update the possible grid list
            return []

        # Precompute valid row combinations, considering known values
        def ValidRowCombinations(RowSum, Zeros, Length, MaxVal, KnownValues):
            """
            Generate all valid row combinations that match constraints and respect known values.
            """
            ValidCombinations = []
            for Comb in itertools.product(range(MaxVal + 1), repeat=Length):
                if sum(Comb) == RowSum and Comb.count(0) == Zeros:
                    # Check known values** before adding
                    if all((KnownValues[i] == -1 or KnownValues[i] == Comb[i]) for i in range(Length)):
                        ValidCombinations.append(Comb)
            return ValidCombinations

        # Generate valid rows and columns based on constraints
        ValidRows = [ValidRowCombinations(YPointsSubset[i], YVoltorbesSubset[i], Cols, MaxValue, MatrixDataTiles[i]) for i in range(Rows)]

        # Compute the number of combinations
        NCombinations = np.prod([len(ValidRows[i]) for i in range(Rows)])

        return ValidRows, NCombinations

    def CmptGridsGenerationOLD(self):
        """
        Generate all plausible matrices based on the given constraints and the current knowledge.
        Returns:
            List of plausible matrices satisfying the constraints.
        """
        Rows, Cols = self.getNRows, self.getNCols
        MaxValue = self.getMaxVal
        PlausibleMatrices = []
        MatrixDataTiles = self.getMatrixDataTiles  # Known tiles are used to filter valid rows (-1 are unknown tiles))

        # Dynamically slice the Points and Voltorbes vectors to match the current grid size
        XPointsSubset = self.getXPoints[:Cols].flatten()  # Take the first `Cols` elements
        YPointsSubset = self.getYPoints[:Rows].flatten()  # Take the first `Rows` elements
        XVoltorbesSubset = self.getXVoltorbes[:Cols].flatten()  # Take the first `Cols` elements
        YVoltorbesSubset = self.getYVoltorbes[:Rows].flatten()  # Take the first `Rows` elements

        # Verify that the sum of the points and voltorbes in X and Y vectors match
        if np.sum(YPointsSubset) != np.sum(XPointsSubset) or np.sum(YVoltorbesSubset) != np.sum(XVoltorbesSubset):
            print("Error: Different total number of points or voltorbes on the whole matrix.")
            self.getLPossibleGrid = []  # Update the possible grid list
            return []

        # Precompute valid row combinations
        def ValidRowCombinations(RowSum, Zeros, Length, MaxVal):
            """
            Generate all valid row combinations given constraints.
            """
            ValidCombinations = []
            for Comb in itertools.product(range(MaxVal + 1), repeat=Length):
                if sum(Comb) == RowSum and Comb.count(0) == Zeros:
                    ValidCombinations.append(Comb)
            return ValidCombinations

        # Generate valid rows and columns based on constraints
        ValidRows = [ValidRowCombinations(YPointsSubset[i], YVoltorbesSubset[i], Cols, MaxValue) for i in range(Rows)]

        # Recursive function to construct matrices row by row
        def ConstructMatrix(Matrix, RowIdx):
            if RowIdx == Rows:
                # Validate column constraints
                if (
                    np.all(np.sum(Matrix, axis=0) == XPointsSubset) and
                    np.all(np.sum(Matrix == 0, axis=0) == XVoltorbesSubset)
                ):
                    PlausibleMatrices.append(Matrix.copy())
                return

            for Row in ValidRows[RowIdx]:
                Matrix[RowIdx, :] = Row
                ConstructMatrix(Matrix, RowIdx + 1)

        # Initialize an empty matrix and start backtracking
        matrix = np.zeros((Rows, Cols), dtype=int)
        ConstructMatrix(matrix, 0)

        self.getLPossibleGrid = PlausibleMatrices  # Update the possible grid list
        return PlausibleMatrices

    def CmptGridsGeneration(self):
        """
        Generate all plausible matrices based on the given constraints and the current knowledge.
        Returns:
            List of plausible matrices satisfying the constraints.

        
        """
        Rows, Cols = self.getNRows, self.getNCols
        MaxValue = self.getMaxVal
        PlausibleMatrices = []
        MatrixDataTiles = self.getMatrixDataTiles  # Known tiles are used to filter valid rows (-1 are unknown tiles))

        # Dynamically slice the Points and Voltorbes vectors to match the current grid size
        XPointsSubset = self.getXPoints[:Cols].flatten()  # Take the first `Cols` elements
        YPointsSubset = self.getYPoints[:Rows].flatten()  # Take the first `Rows` elements
        XVoltorbesSubset = self.getXVoltorbes[:Cols].flatten()  # Take the first `Cols` elements
        YVoltorbesSubset = self.getYVoltorbes[:Rows].flatten()  # Take the first `Rows` elements

        # Verify that the sum of the points and voltorbes in X and Y vectors match
        if np.sum(YPointsSubset) != np.sum(XPointsSubset) or np.sum(YVoltorbesSubset) != np.sum(XVoltorbesSubset):
            print("Error: Different total number of points or voltorbes on the whole matrix.")
            self.getLPossibleGrid = []  # Update the possible grid list
            return []

        # Generate valid rows and columns based on constraints
        ValidRows, NCombinations = self.CmptRowCombinations()

        # Recursive function to construct matrices row by row
        def ConstructMatrix(Matrix, RowIdx):
            if RowIdx == Rows:
                # Validate column constraints
                if (
                    np.all(np.sum(Matrix, axis=0) == XPointsSubset) and
                    np.all(np.sum(Matrix == 0, axis=0) == XVoltorbesSubset)
                ):
                    PlausibleMatrices.append(Matrix.copy())
                return

            for Row in ValidRows[RowIdx]:
                Matrix[RowIdx, :] = Row
                ConstructMatrix(Matrix, RowIdx + 1)

        # Initialize an empty matrix and start backtracking
        matrix = np.zeros((Rows, Cols), dtype=int)
        ConstructMatrix(matrix, 0)

        self.getLPossibleGrid = PlausibleMatrices  # Update the possible grid list
        return PlausibleMatrices

    def CmptGridsGenerationTemp(self):
        """
        Generate all plausible matrices based on the given constraints and the current knowledge.
        Returns:
            List of plausible matrices satisfying the constraints.

        Based on computation of both valid rows and columns.
        """
        Rows, Cols = self.getNRows, self.getNCols
        MaxValue = self.getMaxVal
        MatrixDataTiles = self.getMatrixDataTiles  # Known values (-1 means unknown)

        # Extract constraints
        XPointsSubset = self.getXPoints[:Cols].flatten()  # Take the first `Cols` elements
        YPointsSubset = self.getYPoints[:Rows].flatten()  # Take the first `Rows` elements
        XVoltorbesSubset = self.getXVoltorbes[:Cols].flatten()  # Take the first `Cols` elements
        YVoltorbesSubset = self.getYVoltorbes[:Rows].flatten()  # Take the first `Rows` elements

        # Ensure point and voltorbe constraints match
        if np.sum(YPointsSubset) != np.sum(XPointsSubset) or np.sum(YVoltorbesSubset) != np.sum(XVoltorbesSubset):
            print("Error: Mismatch in total points or voltorbes across rows and columns.")
            self.getLPossibleGrid = []
            return []

        # Generate valid row combinations while respecting known values
        def ValidCombinations(TargetSum, ZeroCount, Length, MaxVal, KnownValues):
            """
            Generate all valid row/column combinations that match constraints and respect known values.
            """
            ValidCombinations = []
            for Comb in itertools.product(range(MaxVal + 1), repeat=Length):
                if sum(Comb) == TargetSum and Comb.count(0) == ZeroCount:
                    # Ensure known values are correctly placed
                    if all((KnownValues[i] == -1 or KnownValues[i] == Comb[i]) for i in range(Length)):
                        ValidCombinations.append(Comb)
            return ValidCombinations

        # Precompute valid row and column candidates using constraints
        ValidRows = [
            ValidCombinations(YPointsSubset[i], YVoltorbesSubset[i], Cols, MaxValue, MatrixDataTiles[i])
            for i in range(Rows)
        ]
        ValidCols = [
            ValidCombinations(XPointsSubset[j], XVoltorbesSubset[j], Rows, MaxValue, MatrixDataTiles[:, j])
            for j in range(Cols)
        ]

        # Recursive function to construct matrices row by row
        def ConstructMatrix(Matrix, RowIdx):
            if RowIdx == Rows:
                # Verify column constraints before accepting the matrix
                if all(tuple(Matrix[:, j]) in ValidCols[j] for j in range(Cols)):
                    PlausibleMatrices.append(Matrix.copy())  # Save fully constructed matrices
                return

            for row in ValidRows[RowIdx]:
                Matrix[RowIdx, :] = row  # Assign valid row
                ConstructMatrix(Matrix, RowIdx + 1)

        # Initialize an empty matrix without `-1`
        PlausibleMatrices = []
        matrix = np.zeros((Rows, Cols), dtype=int)  # Start with 0s
        ConstructMatrix(matrix, 0)

        self.getLPossibleGrid = PlausibleMatrices  # Update possible grid list
        return PlausibleMatrices

    def CmptElem2MatricesSelection(self, Row, Col, Val):
        """
        Update the possible matrices by assigning a specific Val to an element.

        Args:
            Row (int): The Row index of the element.
            Col (int): The Column index of the element.
            Val (int): The Val to assign to the element.

        Returns:
            None
        """
        if not (0 <= Row < self.getNRows and 0 <= Col < self.getNCols):
            print("Error: Invalid position ({Row}, {Col}). Must be within grid dimensions.")

        # Filter plausible matrices
        UpdatedMatrices = []
        for Matrix in self.getLPossibleGrid:
            if Matrix[Row, Col] == Val:
                UpdatedMatrices.append(Matrix)

        # Check if no matrices are left after the update
        if not UpdatedMatrices:
            print("Warning: No plausible matrices remain after the update. Check the input values.")
            return False

        # Update the list of plausible matrices
        self.getLPossibleGrid = UpdatedMatrices
        self.AddTile2MatrixDataTiles(Val=Val, Row=Row, Column=Col)
        return True

    def CmptProba(self):
        """
        Calculate the probability of finding a 0 at each position in the grid
        based on the list of possible grids.
        Returns:
            A 2D NumPy array representing the probability of a 0 at each grid position.
        """
        if not self.getLPossibleGrid:
            print("Error: LPossibleGrid is empty.")
            return None
        # Stack all grids into a 3D NumPy array for efficient computation
        MatricesStack = np.array(self.getLPossibleGrid)  # Shape: (num_grids, Rows, Cols)

        # Compute the probability of 0 at each position
        PMatrix = np.mean(MatricesStack == 0, axis=0)*100

        self.getMatrixProb = PMatrix

        return PMatrix

    # Display methods
    def PLTDataTiles(self, paramPLT=None):
        if paramPLT is None:
            paramPLT = DefaultParamPLT()

        if self.getMatrixDataTiles is None:
            print("Error: MatrixDataTiles is empty.")
            return

        PLTImShow(ValMatrix=self.getMatrixDataTiles, paramPLT=paramPLT, FInterpolType=2, BShowVal=True, Fmt=".0f")
        paramPLT.getTitle = "Values of known tiles"
        paramPLT.getXLabel = "Columns"
        paramPLT.getYLabel = "Rows"
        PLTShow(paramPLT=paramPLT)
        return paramPLT

    def PLTProba(self, paramPLT=None):
        if paramPLT is None:
            paramPLT = DefaultParamPLT()

        if self.getMatrixProb is None:
            print("Error: MatrixProb is empty.")
            return

        paramPLT.getTitle = "Probability of 0 at each position"
        paramPLT.getXLabel = "Columns"
        paramPLT.getYLabel = "Rows"
        PLTImShow(ValMatrix=self.getMatrixProb, paramPLT=paramPLT, FInterpolType=2)
        PLTShow(paramPLT=paramPLT)
        return paramPLT

    def PLTCmptScores(self, paramPLT=None):
        if paramPLT is None:
            paramPLT = DefaultParamPLT()

        if self.getMatrixCmptScores is None:
            print("Error: MatrixCmptScores is empty.")

        paramPLT.getTitle = "Computed scores for each position"
        paramPLT.getXLabel = "Columns"
        paramPLT.getYLabel = "Rows"
        PLTImShow(ValMatrix=self.getMatrixCmptScores, paramPLT=paramPLT, FInterpolType=2)
        PLTShow(paramPLT=paramPLT)
        return paramPLT

    def PLTProbaAndTiles(self, paramPLT=None, BStartPLT=True):
        if paramPLT is None:
            paramPLT = DefaultParamPLT()

        Index = PLTMultiPlot(paramPLT, Rows=1, Cols=2, BStartPLT=BStartPLT)
        self.PLTProba(paramPLT=paramPLT)

        Index = PLTMultiPlot(paramPLT, Rows=1, Cols=2, Index=Index)
        self.PLTDataTiles(paramPLT=paramPLT)

        Index = PLTMultiPlot(paramPLT, Rows=1, Cols=2, Index=Index)
        return paramPLT

    def PLTProbaScoresTiles(self, paramPLT=None, BStartPLT=True):
        if paramPLT is None:
            paramPLT = DefaultParamPLT()

        Index = PLTMultiPlot(paramPLT, Rows=1, Cols=3, BStartPLT=BStartPLT)
        self.PLTProba(paramPLT=paramPLT)

        Index = PLTMultiPlot(paramPLT, Rows=1, Cols=3, Index=Index)
        self.PLTCmptScores(paramPLT=paramPLT)

        Index = PLTMultiPlot(paramPLT, Rows=1, Cols=3, Index=Index)
        self.PLTDataTiles(paramPLT=paramPLT)

        Index = PLTMultiPlot(paramPLT, Rows=1, Cols=3, Index=Index)
        return paramPLT

    # Solver methods
    def VoltorbGameSolver(self, NumMethod=0):
        """
        Solve the Voltorb game using the specified method.

        Policy:
        Request the value of the reward at a given place (least probability),
        recompute the probability matrix from this knowledge, and show the new 
        probability matrix.

        Improvements:
        - Add the simplification of management of asking tiles values
        - Add the possibility to compute the number of combinations with row or with column constraints and choose the one with the least number of combinations
        """
        Value = True
        Status = True
        i = 1
        while Status:
            # Compute the number of grids that satisfies only row constraints
            print(f"Step {i}:")
            ValidRows, NCombinations = self.CmptRowCombinations()
            print(f"Number of combination {NCombinations}")

            # Verify if the number of combinations is not too high
            if NCombinations > 10**6:
                print("Info: Too many combinations to compute. Solver will not work. Use the score method instead.")

                # Compute scores
                self.CmptScores()

                # Retrieve the computed scores matrix
                ScoreMatrix = self.getMatrixCmptScores

                # Retrieve the data matrix
                DataMatrix = self.getMatrixDataTiles

                # Plottings of the probability, scores and data matrices
                paramPLT = self.PLTProbaScoresTiles(paramPLT=None, BStartPLT=True)
                PLTScreenMaximize()

                # If no position has a score higher than 1, the game is finished
                if np.all(ScoreMatrix <= 1):
                    Status = False
                    print("Info: Solver completed successfully.")
                    break

                # If there are infinite values in the score matrix, the player has to select those tiles
                if np.any(ScoreMatrix == np.inf):
                    # Select the tile with the infinite score
                    Row, Col = np.unravel_index(np.argmax(ScoreMatrix), ScoreMatrix.shape)

                    # Ask for the value at the selected position
                    print(f"Querying value at position ({Row}, {Col})")
                    Value = int(input())
                else:
                    # If tiles are known, the solver will not select them
                    # Ask for the row and column of the tiles
                    TempStatus = True
                    while TempStatus:
                        print("Querying the row of the tile")
                        Row = int(input())
                        print("Querying the column of the tile")
                        Col = int(input())
                        # As long as the position is invalid, the player has to give a valid position
                        if Row < 0 or Row >= self.getNRows or Col < 0 or Col >= self.getNCols:
                            print("Error: Invalid position.")
                        else:
                            TempStatus = False
                    # Ask for the value at the selected position
                    print(f"Querying value at position ({Row}, {Col})")
                    Value = int(input())

                if Value == 0:
                    Status = False
                    print("Info: Terminating solver as value is 0 or False.")
                else:
                    # Update the plausible matrices with the new knowledge
                    self.AddTile2MatrixDataTiles(Val=Value, Row=Row, Column=Col)

            else:
                # End of the score method and goes to the score
                Status = False
                print("Info: Solver use now the score and probability method.")

            # Increment the step counter
            i += 1

        if Value != 0:  # Launch the solver
            # Generate plausible matrices
            self.CmptGridsGeneration()

            Status = True
            while Status:
                # Compute probabilities
                self.CmptProba()

                # Compute scores
                self.CmptScores()

                # Retrieve the computed scores matrix
                ScoreMatrix = self.getMatrixCmptScores

                # Retrieve the data matrix
                DataMatrix = self.getMatrixDataTiles

                # Retrieve the probability matrix
                ProbMatrix = self.getMatrixProb

                # If no position has a score higher than 1, the game is finished
                if np.all(ScoreMatrix <= 1):
                    Status = False
                    print("Info: Solver completed successfully.")
                    break

                # If tiles are known, the solver will not select them
                TempProbMatrix = np.where(DataMatrix <= 0, ProbMatrix, np.inf)
                # If the score is less than 1, the tile is a voltorbe or a 1 point tile, the solver will not select them
                TempProbMatrix = np.where(ScoreMatrix < 1,  np.inf, TempProbMatrix)
                # From the previous selection, the solver will select the tile with the lowest probability
                Row, Col = np.unravel_index(np.argmin(TempProbMatrix), TempProbMatrix.shape)

                # Display the current state
                print(f"Step {i}:")
                print(f"Number of plausible matrices: {len(self.getLPossibleGrid)}")
                print(f"Querying value at position ({Row}, {Col})")

                # Plottings of the probability, scores and data matrices
                paramPLT = self.PLTProbaScoresTiles(paramPLT=None, BStartPLT=True)
                PLTScreenMaximize()

                # Ask for the value at the selected position
                Value = int(input())

                if Value == 0:
                    Status = False
                    print("Info: Terminating solver as value is 0 or False.")
                elif len(self.getLPossibleGrid) == 1:  # Assuming checkCompletion verifies if the task is finished
                    Status = False
                    print("Info: Solver completed successfully (only one possibility).")
                else:
                    # Update the plausible matrices with the new knowledge
                    Success = self.CmptElem2MatricesSelection(Row, Col, Value)
                    if not Success:
                        Status = True
                        print("Error: Wrong value")

                # Increment the step counter
                i += 1

        # Close all plots
        CloseALLPlots()





