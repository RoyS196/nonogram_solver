import numpy as np
import logging
from itertools import combinations
from queue import PriorityQueue, Empty

from nonogram import LineClues, Nonogram, NonogramGrid


def check_potential_line(potential_line: np.ndarray, current_line: np.ndarray) -> bool:
    """Checks a potential line against a current line. Returns True if lines are compatible, returns False if the
    potential line is not possible given the current line. The potential line is not compatible with the current line
    ONLY IF any of its values unequal to -1 are not identical to the corresponding values of the current line AND the
    corresponding values of the current line are not -1.

    Args:
        potential_line: the line (1d array) which is checked against current_line.
        current_line: the line (1d array) against which potential_line is checked.

    Returns:
        boolean corresponding to whether potential_line is possible given current_line.
    """
    comparison_line = current_line.copy()
    comparison_line[comparison_line == -1] = potential_line[comparison_line == -1]

    # Also do a "reverse check" in case potential line is not a complete line
    potential_line_copy = potential_line.copy()
    potential_line_copy[potential_line_copy == -1] = comparison_line[potential_line_copy == -1]
    return np.array_equal(potential_line_copy, comparison_line)


# TODO: update current_line or return new line
def update_current_line(new_line: np.ndarray, current_line: np.ndarray) -> None:
    """Updates current line based on new information from new line, if the new line is compatible with the current
    line. It does so by only setting all values of -1 of the current line to the corresponding values of the new line,
    ignoring all values unequal to -1 of the current line.

    Args:
        new_line: the line (1d array) that is used to update current_line.
        current_line: the line (1d array) that is updated.

    Returns:
        None

    Raises:
        ValueError: if new_line contradicts current_line.
    """
    if check_potential_line(potential_line=new_line, current_line=current_line):
        current_line[current_line == -1] = new_line[current_line == -1]
    else:
        msg = "new_line not compatible with current_line. Cannot update current_line."
        logging.error(msg)
        raise ValueError(msg)


def obtain_lines_intersection(list_lines: list[np.ndarray]) -> np.ndarray:
    """Obtains an intersection line from a list of equally sized lines. If all lines from the list have an identical
    value at a specific position, the returned intersection line has this values at the corresponding position.
    Elsewhere this returned intersection line has value -1.

    Args:
        list_lines: list of lines (1d arrays) from which an intersection line is returned.

    Returns:
        intersection line (1d array).
    """
    potential_squares_array = np.array(list_lines).T
    line_length = potential_squares_array.shape[0]
    intersection_line = np.full((line_length,), -1)

    for i, potential_squares in enumerate(potential_squares_array):
        if np.all(potential_squares == potential_squares[0]):
            intersection_line[i] = potential_squares[0]

    return intersection_line


def line_leeway_method(line_clues: LineClues, current_line: np.ndarray) -> None:
    """Updates current line based on line clues using the leeway method. The 'leeway' of a block/line is the number of
    squares can we move a block, based only on the line clues and line length. If the block is larger then its leeway
    squares can be filled in, without any other information about the grid. To update the line based this method loops
    twice through all blocks (once in reverse) to determine the squares that can be filled in.

    Args:
        line_clues: LineClues object, that contains the block sizes, block colors and line length.
        current_line: line (1d array) that is updated using the leeway method.

    Returns:
        None
    """
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

        # fill in subblocks if start<end
        block_colors = line_clues.block_colors
        for i, subblock_start in enumerate(subblock_start_list):
            new_line[subblock_start:subblock_end_list[i]] = block_colors[i]

        # no leeway, since the first value is not -1, hence fill in separating 0's
        if new_line[0] != -1:
            new_line[new_line == -1] = 0

    update_current_line(new_line=new_line, current_line=current_line)


def grid_leeway_method(nonogram: Nonogram, current_grid: NonogramGrid) -> None:
    """Updates nonogram grid by applying line_leeway_method on every row/column of nonogram grid, using all line clues
    of a nonogram.

    Args:
        nonogram: Nonogram object containing all LineClues object of nonogram
        current_grid: NonogramGrid object containing the nonogram grid (np array) that is updated

    Returns:
        None
    """
    n_row, n_col = nonogram.n_row, nonogram.n_col

    for i in range(0, n_row):
        current_row = current_grid.get_row(i)
        row_clues = nonogram.get_row_clues(i)
        line_leeway_method(line_clues=row_clues, current_line=current_row)

    for j in range(0, n_col):
        current_col = current_grid.get_col(j)
        col_clues = nonogram.get_col_clues(j)
        line_leeway_method(line_clues=col_clues, current_line=current_col)


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
