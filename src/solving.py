from nonogram import LineClues, Nonogram, NonogramGrid
from itertools import combinations
import numpy as np
import logging
from queue import PriorityQueue, Empty


# Checks potential line (does not need to be complete) against current line. Returns true if potential line is possible, false if not
def check_potential_line(potential_line: np.ndarray, current_line: np.ndarray) -> bool:
    comparison_line = current_line.copy()
    comparison_line[comparison_line == -1] = potential_line[comparison_line == -1]

    # Also do a "reverse check" in case potential line is not a complete line
    potential_line_copy = potential_line.copy()
    potential_line_copy[potential_line_copy == -1] = comparison_line[potential_line_copy == -1]
    return np.array_equal(potential_line_copy, comparison_line)


# TODO: update current_line or return new line
def update_current_line(new_line: np.ndarray, current_line: np.ndarray) -> None:
    # TODO: might be unecessary to check line
    if check_potential_line(potential_line=new_line, current_line=current_line):
        current_line[current_line == -1] = new_line[current_line == -1]
    else:
        msg = "new_line not compatible with current_line. Cannot update current_line."
        logging.error(msg)
        raise ValueError(msg)


# TODO: function description
def obtain_lines_intersection(list_lines: list[np.ndarray]) -> np.ndarray:
    potential_squares_array = np.array(list_lines).T
    line_length = potential_squares_array.shape[0]
    intersection_line = np.full((line_length,), -1)

    for i, potential_squares in enumerate(potential_squares_array):
        if np.all(potential_squares == potential_squares[0]):
            intersection_line[i] = potential_squares[0]

    return intersection_line


# TODO: function description
def line_leeway_method(line_clues: LineClues, current_line: np.ndarray) -> None:
    logging.debug(f"FUNCTION: line_leeway_method, line_clues: {line_clues}, current_line: {current_line}")
    line_length = line_clues.line_length
    new_line = np.full((line_length,), -1)

    # If no clues, update all values to zeroes
    if line_clues.is_empty():
        new_line[:] = 0

    else:
        # loop through block in normal order to determine the start of subblocks
        subblock_end_list = []
        subblock_end = 0
        prev_block_color = 0
        for block_size, block_color in line_clues:
            if block_color == prev_block_color:
                subblock_end += 1

            subblock_end += block_size
            subblock_end_list.append(subblock_end)
            prev_block_color = block_color

        subblock_start_list = []
        subblock_start = 0
        prev_block_color = 0
        for block_size, block_color in reversed(line_clues):
            if block_color == prev_block_color:
                subblock_start -= 1

            subblock_start -= block_size
            subblock_start_list.append(subblock_start)
            prev_block_color = block_color
        subblock_start_list.reverse()

        block_colors = line_clues.block_colors
        for i, subblock_start in enumerate(subblock_start_list):
            new_line[subblock_start:subblock_end_list[i]] = block_colors[i]

        if new_line[0] != -1:  # no leeway, since the first square is filled in (not -1)
            new_line[new_line == -1] = 0

    update_current_line(new_line=new_line, current_line=current_line)


# Everything below: no tests yet
# # TODO: function description
# def grid_leeway_method(nonogram: Nonogram, current_grid: NonogramGrid) -> None:
#     n_row, n_col = nonogram.n_row, nonogram.n_col

#     for i in range(0, n_row):
#         current_row = current_grid.get_row(i)
#         row_clues = nonogram.get_row_clues(i)
#         line_leeway_method(line_clues=row_clues, current_line=current_row)

#     for j in range(0, n_col):
#         current_col = current_grid.get_col(j)
#         col_clues = nonogram.get_col_clues(j)
#         line_leeway_method(line_clues=col_clues, current_line=current_col)


# # TODO: function description
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
