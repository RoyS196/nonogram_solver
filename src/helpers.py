import numpy as np
from nonogram import LineClues, Nonogram, NonogramGrid


def obtain_grid_from_csv(filename: str, delimiter: str) -> NonogramGrid:
    grid_array = np.genfromtxt(fname=filename, delimiter=delimiter, dtype=np.integer, filling_values=-1)
    return NonogramGrid(grid_array=grid_array)


def obtain_nonogram_from_grid() -> Nonogram:
    pass
