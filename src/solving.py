import numpy as np
import logging
from itertools import combinations
from queue import PriorityQueue, Empty

from nonogram import LineClues, Nonogram, NonogramGrid


def check_arrays_compatibility(array_1: np.ndarray, array_2: np.ndarray) -> bool:
    """Returns True if numpy arrays are 'compatible', returns False otherwise.

    The arrays are compatible if and only if all values of array_1 are equal to the corresponding values of array_2,
    ignoring all values that are equal to -1 in one of the two arrays.

    Args:
        array_1 (np.ndarray): the numpy array which is checked against array_2.
        array_2 (np.ndarray): the numpy array which is checked against array_1.

    Returns:
        A boolean corresponding to whether the arrays are compatible.

    Raises:
        ValueError: if the shape of array_1 and array_2 are not equal.
    """
    # Copy arrays to prevent mutating the original
    array_1_copy = array_1.copy()
    array_2_copy = array_2.copy()

    # Change every value of -1 to the corresponding value of the other array
    array_1_copy[array_1_copy == -1] = array_2[array_1_copy == -1]
    array_2_copy[array_2_copy == -1] = array_1[array_2_copy == -1]

    return np.array_equal(array_1_copy, array_2_copy)


# TODO: update current_line or return new line
def update_array_values(base_array: np.ndarray, array_update: np.ndarray) -> None:
    """Updates all values equal to -1 of numpy array to corresponding values of other numpy array.

    Args:
        base_array (np.ndarray): the array whose values are updated.
        array_update: the array from which values are used to update array.

    Returns:
        None

    Raises:
        ValueError: if the shape of base_array and array_update are not equal.
        ValueError: if base_array and array_update are not compatible based on check_arrays_compatibility function.
    """
    if not base_array.shape == array_update.shape:
        msg = f"""Cannot update base_update. Shape of base_array {base_array.shape} not equal to shape of array_update
        {array_update.shape}."""
        logging.error(msg)
        raise ValueError(msg)
    if not check_arrays_compatibility(array_1=base_array, array_2=array_update):
        msg = "Cannot update base_update. base_array not compatible with array_update."
        logging.error(msg)
        raise ValueError(msg)

    base_array[base_array == -1] = array_update[base_array == -1]


def obtain_arrays_intersection(list_arrays: list[np.ndarray]) -> np.ndarray:
    """Obtains a numpy (intersection) array from a list of equally sized numpy arrays.

    If a value at a specific position is equal for all arrays in the list, then the corresponding value in the
    intersection array is equal to this value. All other values in the intersection array are equal to -1.

    Args:
        list_arrays (list[np.ndarray]): list of arrays from which an intersection array is returned.

    Returns:
        A numpy array with values -1 or values which are equal for all arrays in list_arrays.

    Raises:
        ValueError: if the shapes of all arrays in list_arrays are not equal.
    """
    if not all(list_arrays[0].shape == array.shape for array in list_arrays):
        msg = "Shapes of all arrays in list_arrays are not equal."
        logging.error(msg)
        raise ValueError(msg)

    comparison_array = np.array(list_arrays).T
    line_length = comparison_array.shape[0]
    intersection_array = np.full((line_length,), -1)

    for i, potential_squares in enumerate(comparison_array):
        if np.all(potential_squares == potential_squares[0]):
            intersection_array[i] = potential_squares[0]

    return intersection_array


def line_leeway_solving_method(line_array: np.ndarray, line_clues: LineClues) -> None:
    """Updates line (1d) array based on nonogram line clues using the leeway solving method.

    The 'leeway' of a block/line in a nonogram is the number of squares a block can move, based only on the line clues
    and line length. If the block is larger then its leeway squares can be filled in, without any other information
    about the grid.

    Args:
        line_array (np.ndarray): line array that is updated using the leeway solving method.
        line_clues (LineClues): object that contains the block sizes, block colors and line length.

    Returns:
        None

    Raises:
        ValueError: if the line_array is not a 1-dimensional array
    """
    if line_array.ndim != 1:
        msg = "line_array is not a 1-dimensional array."
        logging.error(msg)
        raise ValueError(msg)

    line_length = line_clues.line_length
    new_line = np.full((line_length,), -1)

    # If there are no blocks in the line, the updated line should be empty
    if line_clues.is_empty():
        new_line[:] = 0

    # Fill in subblocks if blocks larger then leeway: line_length - sum of blocks - no. necessary 0's seperating blocks
    else:
        subblock_end_list = []
        subblock_end = 0
        prev_block_color = 0

        # loop through blocks in normal order to determine the end of the subblocks
        for block_size, block_color in line_clues:
            # If successive blocks are the same color, there is a necessary separating 0
            if block_color == prev_block_color:
                subblock_end += 1

            subblock_end += block_size
            subblock_end_list.append(subblock_end)
            prev_block_color = block_color

        subblock_start_list = []
        subblock_start = 0
        prev_block_color = 0

        # loop through blocks in reverse order to determine the start of the subblocks
        for block_size, block_color in reversed(line_clues):
            # If successive blocks are the same color, there is a necessary separating 0
            if block_color == prev_block_color:
                subblock_start -= 1
            subblock_start -= block_size
            subblock_start_list.append(subblock_start)
            prev_block_color = block_color

        subblock_start_list.reverse()

        # TODO: merge 3rd loop in one of the other loops
        # fill in subblocks if start<end
        block_colors = line_clues.block_colors
        for i, subblock_start in enumerate(subblock_start_list):
            new_line[subblock_start:subblock_end_list[i]] = block_colors[i]

        # no leeway, since the first value is not -1, hence fill in separating 0's
        if new_line[0] != -1:
            new_line[new_line == -1] = 0

    update_array_values(base_array=line_array, array_update=new_line)


def grid_leeway_solving_method(current_grid: NonogramGrid, nonogram: Nonogram) -> None:
    """Updates nonogram grid using the leeway solving method on every row/column of nonogram grid.

    Args:
        nonogram (Nonogram): object containing all LineClues objects for all rows/columns
        current_grid (NonogramGrid): object containing the nonogram grid_array that is updated.

    Returns:
        None
    """
    n_row, n_col = nonogram.n_row, nonogram.n_col

    for i in range(0, n_row):
        current_row = current_grid.get_row(i)
        row_clues = nonogram.get_row_clues(i)
        line_leeway_solving_method(line_array=current_row, line_clues=row_clues)

    for j in range(0, n_col):
        current_col = current_grid.get_col(j)
        col_clues = nonogram.get_col_clues(j)
        line_leeway_solving_method(line_array=current_col, line_clues=col_clues)


# Everything below: no tests yet
# def calculate_initial_line_priority(line_clues: LineClues, current_line: np.ndarray = None) -> float:
#     priority = line_clues.line_length - line_clues.min_length  # leeway
#     if current_line is not None:
#         pass  # TODO: take current line into account

#     return priority


# # TODO: function description
# def calculate_initial_priority_queue(nonogram: Nonogram, current_grid: NonogramGrid = None) -> PriorityQueue:
#     n_row, n_col = nonogram.n_row, nonogram.n_col
#     priority_queue = PriorityQueue()

#     for i in range(0, n_row):
#         current_row = current_grid.get_row(i)
#         row_clues = nonogram.get_row_clues(i)
#         priority = calculate_initial_line_priority(line_clues=row_clues, current_line=current_row)
#         priority_queue.put((priority, "row", i))

#     for j in range(0, n_col):
#         current_col = current_grid.get_col(j)
#         col_clues = nonogram.get_col_clues(j)
#         priority = calculate_initial_line_priority(line_clues=col_clues, current_line=current_col)
#         priority_queue.put((priority, "col", j))

#     return priority_queue


# # TODO: method description
# def determine_line_combinations(line_clues_object: LineClues, current_line=None) -> list[np.ndarray]:
#     check_against_current_line = current_line is not None
#     potential_lines = []

#     # obtain values from LineClues object
#     block_sizes = line_clues_object.block_sizes
#     block_colors = line_clues_object.block_colors
#     leeway = line_clues_object.leeway
#     n_blocks = line_clues_object.n_blocks
#     line_length = line_clues_object.line_length

#     # Determine all potential lines using combinations
#     for combination in combinations(range(0, leeway + n_blocks), n_blocks):
#         potential_line = np.full((line_length,), 0)
#         sum_of_blocks = 0
#         for i, c in enumerate(combination):
#             potential_line[sum_of_blocks + c:sum_of_blocks + c + line_clues_object.get_block_size(i)] = block_colors[i]
#             sum_of_blocks += block_sizes[i]
#         if check_against_current_line:
#             if check_potential_line(potential_line=potential_line, current_line=current_line):
#                 potential_lines.append(potential_line)
#         else:
#             potential_lines.append(potential_line)

#     return potential_lines


# # Creates two initial listsof possible row/columns, given initial grid (if no initial grid, do not check)
# def initialize_potential_lines(nonogram: Nonogram, current_grid: NonogramGrid) -> tuple[list[np.ndarray], list[np.ndarray]]:
#     n_row = nonogram.n_row
#     n_col = nonogram.n_col

#     potential_rows = []
#     for i in range(0, n_row):
#         current_row = None
#         if current_grid is not None:
#             current_row = current_grid.get_row(i)
#         potential_rows.append(determine_line_combinations(line_clues_object=nonogram.row_clues_tuple[i], current_line=current_row))
#     potential_rows = tuple(potential_rows)

#     potential_cols = []
#     for j in range(0, n_col):
#         current_line = None
#         if current_grid is not None:
#             current_line = current_grid.get_col(j)
#         potential_cols.append(determine_line_combinations(line_clues_object=nonogram.col_clues_tuple[i], current_line=current_line))
#     potential_cols = tuple(potential_cols)

#     return potential_rows, potential_cols


# # TODO: function description
# def queue_solution_method(nonogram: Nonogram, current_grid: NonogramGrid = None, prio_queue: PriorityQueue = None) -> None:
#     pass  # TODO


if __name__ == "__main__":
    pass

    # Step 1: empty line shuffle, and create initial queue

    # Step 2: use queue for queue algorithm

    # Step 3: solve remainder

    # Step 4: return (partial) solution
