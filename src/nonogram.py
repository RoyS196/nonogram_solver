import logging
import numpy as np


# TODO: class description, simplify?
class LineClues:  # TODO: lists -> tuple
    def __init__(self, block_sizes: tuple[int], line_length: int, block_colors: tuple[int] = None):
        if isinstance(block_sizes, int):
            block_sizes = (block_sizes,)

        if isinstance(block_colors, int):
            block_colors = (block_colors,)

        if block_colors is None:
            block_colors = (1,) * len(block_sizes)

        if not all(isinstance(i, (int, np.integer)) for i in block_sizes):
            msg = "line_clues argument must be tuple of integers"
            logging.error(msg)
            raise TypeError(msg)

        if not all(isinstance(i, (int, np.integer)) for i in block_colors):
            msg = "line_colors optional argument must be tuple of integers"
            logging.error(msg)
            raise TypeError(msg)

        if len(block_sizes) != len(block_colors):
            msg = "line_clues and block_colors must have the same length"
            logging.error(msg)
            raise TypeError(msg)

        if len(block_sizes) > 0:
            if min(block_sizes) <= 0:
                msg = "line_clues argument must be list of strictly positive integers"
                logging.error(msg)
                raise ValueError(msg)

            if min(block_colors) <= 0:
                msg = "line_colors argument must be list of strictly positive integers"
                logging.error(msg)
                raise ValueError(msg)

        if sum(block_sizes) + len(block_sizes) - 1 > line_length:
            msg = "line_length value smaller then minimal line length implied by line_clues"
            logging.error(msg)
            raise ValueError(msg)

        self.block_sizes = block_sizes
        self.block_colors = block_colors
        self.block_tuples = tuple(zip(block_sizes, block_colors))
        self.n_colors = len(set(block_colors))
        self.n_blocks = len(block_sizes)
        self.line_length = line_length
        self.min_length = sum(block_sizes) + self.n_blocks - 1
        self.leeway = self.line_length - self.min_length

    # TODO: maybe unnecessary
    def __getitem__(self, i):
        return self.block_tuples[i]

    # TODO: maybe unnecessary
    def __iter__(self):
        return iter(self.block_tuples)

    # TODO: maybe unnecessary
    def __len__(self):
        return len(self.block_sizes)

    # TODO: temporary
    def __str__(self):
        if self.n_colors == 1:
            return str(self.block_sizes)
        else:
            return str(self.block_tuples)

    # TODO: temporary
    def __repr__(self):
        return str(self.block_sizes)

    # TODO: function description
    def is_empty(self):
        return self.block_sizes == ()

    # TODO: function description
    def get_block_size(self, i: int) -> int:
        return self.block_sizes[i]

    # TODO: function description
    def get_block_tuple(self, i: int) -> tuple[int, int]:
        return self.block_tuples[i]

    # TODO: function description
    def get_block_color(self, i: int) -> str:
        return self.block_colors[i]


# TODO: class description
class Nonogram:  # TODO: line_clues_list -> line_clues_tuple
    def __init__(self, row_clues_tuple: tuple[LineClues], col_clues_tuple: tuple[LineClues]):

        # TODO: separate row and col checks
        if not all(isinstance(i, LineClues) for i in col_clues_tuple + row_clues_tuple):
            msg = "Arguments must be lists of LineClues classes"
            logging.error(msg)
            raise TypeError(msg)

        self.row_clues_tuple = row_clues_tuple
        self.col_clues_tuple = col_clues_tuple
        self.n_row = len(row_clues_tuple)
        self.n_col = len(col_clues_tuple)
        self.shape = (self.n_row, self.n_col)

        # TODO: make this better (specify which block causes error)
        for row_clues in row_clues_tuple:
            if row_clues.min_length > self.n_col:
                msg = f"A LineClues  ({row_clues}) in row_clues_tuple is too large for puzzle dimensions ({self.shape})"
                logging.error(msg)
                raise ValueError(msg)
        for col_clues in col_clues_tuple:
            if col_clues.min_length > self.n_row:
                msg = f"A LineClues ({col_clues}) in col_clues_tuple is too large for puzzle dimensions ({self.shape})"
                logging.error(msg)
                raise ValueError(msg)

    # TODO: temporary
    def __str__(self):
        return f"Row clues: {str(self.row_clues_tuple)}.\nColumn clues: {str(self.col_clues_tuple)}"

    # TODO: temporary
    def __repr__(self):
        return f"Row clues: {str(self.row_clues_tuple)}.\nColumn clues: {str(self.col_clues_tuple)}"

    # TODO: function description
    def get_row_clues(self, i: int):
        return self.row_clues_tuple[i]

    # TODO: function description
    def get_col_clues(self, j: int):
        return self.col_clues_tuple[j]


# TODO: class description
class NonogramGrid():
    def __init__(self, grid_array: np.ndarray):
        if grid_array is None:
            grid_array

        if not all(isinstance(n, (int, np.integer)) for n in np.unique(grid_array)):
            msg = "Argument must be an array with only integers as values"
            logging.error(msg)
            raise ValueError(msg)

        if np.min(grid_array) < -1:
            msg = "Argument must be an array with only integers greater than or equal to -1 as values"
            logging.error(msg)
            raise ValueError(msg)

        self.grid_array = grid_array
        self.n_row = self.grid_array.shape[0]
        self.n_col = self.grid_array.shape[1]
        self.shape = self.grid_array.shape

    # TODO: temporary
    def __str__(self):
        return str(self.grid_array)

    # TODO: function description
    def __copy__(self):
        return NonogramGrid(grid_array=self.grid_array.copy())

    # TODO: function description
    def set_row(self, i: int, row_array: np.ndarray):
        self.grid_array[i, :] = row_array

    # TODO: function description
    def set_col(self, j: int, col_array: np.ndarray):
        self.grid_array[:, j] = col_array

    # TODO: function description
    def set_value(self, row_col_ij: tuple[int, int], value: int):
        self.grid_array[row_col_ij] = value

    # TODO: function description
    def get_row(self, i: int):
        return self.grid_array[i, :]

    # TODO: function description
    def get_col(self, j: int):
        return self.grid_array[:, j]

    # TODO: function description
    def get_value(self, row_col_ij: tuple[int, int]):
        return self.grid_array[row_col_ij]

    @classmethod
    def instantiate_empty_grid(cls, n_row, n_col):
        grid_array = np.full((n_row, n_col), -1)
        return cls(grid_array=grid_array)
