import numpy as np

from nonogram import LineClues, Nonogram, NonogramGrid


def obtain_grid_from_csv(filename: str) -> NonogramGrid:
    """Obtains a nonogram grid from a csv-file. Empty values in the csv-file correspond to -1 values in the returned
    nonogram grid.

    Args:
        filename: filename of the csv-file

    Returns:
        NonogramGrid object containing the nonogram grid (np array) with values corresponding to the csv-file.
    """
    grid_array = np.genfromtxt(fname=filename, delimiter=",", dtype=np.integer, filling_values=-1)

    return NonogramGrid(grid_array=grid_array)


def obtain_line_clues_from_line(line_array: np.ndarray) -> LineClues:
    """Obtains line clues from a filled in line.

    Args:
        line_array: line (1d array) from which line clues are obtained.

    Returns:
        LineClues object obtained from filled in line.
    """

    # TODO: deal with not completely filled in lines (-1), raise error?
    block_sizes = []
    block_colors = []
    square_counter = 0
    previous_square = 0

    for square in line_array:
        if square != previous_square:
            if previous_square != 0:
                block_sizes.append(square_counter)
                block_colors.append(previous_square)
            square_counter = 0
        square_counter += 1
        previous_square = square

    if previous_square != 0:
        block_sizes.append(square_counter)
        block_colors.append(previous_square)

    return LineClues(block_sizes=tuple(block_sizes), line_length=len(line_array), block_colors=tuple(block_colors))


def obtain_nonogram_from_grid(nonogram_grid: NonogramGrid) -> Nonogram:
    """Obtains nonogram with all line clues from (complete) nonogram grid, by applying obtain_line_clues_from_line to
    each row/col of the nonogram grid.

    Args:
        line_array: line (1d array) from which line clues are obtained.

    Returns:
        Nonogram object containing LineClues objects obtained from filled in nonogram grid.
    """

    # TODO: deal with not completely filled in grid (-1), raise error?
    grid_array = nonogram_grid.grid_array
    row_clues_list = []
    col_clues_list = []

    for row_array in grid_array:
        row_clues_list.append(obtain_line_clues_from_line(row_array))

    for col_array in grid_array.T:
        col_clues_list.append(obtain_line_clues_from_line(col_array))

    return Nonogram(row_clues_tuple=tuple(row_clues_list), col_clues_tuple=tuple(col_clues_list))
