import heapq
import logging
import numpy as np


class LineClues:
    """Class that represents the line clues of a row or column in a nonogram.

    The line clues are the sizes (and colors) of blocks in a specific line (row or column) in a nonogram puzzle. The
    order of the block sizes (and colors) corresponds to the order of the blocks in the specific line in the puzzle
    solution. The class also contains the length of the line."""

    def __init__(self, block_sizes: tuple[int, ...], line_length: int, block_colors: tuple[int, ...] = None) -> None:  # noqa: C901
        """Initializes the line clues class object, given the block sizes (in squares), line length and optionally the
        block colors.

        The block sizes and block colors are passed as tuples of integers >0. The order of block sizes and blocks
        colors represent the order of the blocks in the line: left to right (row) or top to bottom (column). If
        block_colors is not passed, the block colors are all assumed to be the same, represented by the integer 1. The
        tuple of block sizes should be equal to the tuple of block colors (if passed). The line length should be larger
        then the minimal length implied by the block sizes and block colors. Note that successive blocks of the same
        color have a necessary seperating empty square (value/color 0) between them, but blocks of different colors do
        not.

        Args:
            block_sizes (tuple[int]): tuple of integers >0 representing the size of the blocks (number of squares).
            line_length (int): length of the line (row/column), which holds the blocks.
            block_colors (tuple[int], optional): tuple of integers >0 representing the colors of the blocks.

        Raises:
            TypeError: if block_sizes is not an tuple of integers.
            TypeError: if block_colors is not an tuple of integers.
            ValueError: if tuples block_sizes and block_colors are not the same length.
            ValueError: if block_sizes contains a integer <=0
            ValueError: if block_colors contains a integer <=0
            ValueError: if minimal line length implied by block_sizes and block_colors is larger then line_length
        """

        # If block_sizes or block_colors is an integer, convert it to a 1-tuple.
        if isinstance(block_sizes, int):
            block_sizes = (block_sizes,)
        if isinstance(block_colors, int):
            block_colors = (block_colors,)

        # If block_colors is not passed, assume all colors are equal (represented by int 1)
        if block_colors is None:
            block_colors = (1,) * len(block_sizes)

        if not all(isinstance(i, (int, np.integer)) for i in block_sizes):
            msg = "block_sizes argument must be tuple of integers"
            logging.error(msg)
            raise TypeError(msg)

        if not all(isinstance(i, (int, np.integer)) for i in block_colors):
            msg = "block_colors optional argument must be tuple of integers"
            logging.error(msg)
            raise TypeError(msg)

        if len(block_sizes) != len(block_colors):
            msg = "block_sizes and block_colors must have the same length"
            logging.error(msg)
            raise ValueError(msg)

        if len(block_sizes) > 0:
            if min(block_sizes) <= 0:
                msg = "block_sizes argument must be list of strictly positive integers"
                logging.error(msg)
                raise ValueError(msg)

            if min(block_colors) <= 0:
                msg = "block_colors optional argument must be list of strictly positive integers"
                logging.error(msg)
                raise ValueError(msg)

        # Calculate the minimal line length that could contain the blocks, using the sizes and colors
        repeating_colors_count = sum(np.array(block_colors[1:]) == np.array(block_colors[:-1]))
        # Successive repeating colors need a seperating empty square (value/color 0)
        min_length = sum(block_sizes) + repeating_colors_count

        if min_length > line_length:
            msg = f"line_length {(line_length)} value smaller then minimal line length {(min_length)}"
            logging.error(msg)
            raise ValueError(msg)

        self.block_sizes = block_sizes
        self.block_colors = block_colors
        self.block_tuples = tuple(zip(block_sizes, block_colors))
        self.n_unique_colors = len(set(block_colors))
        self.n_blocks = len(block_sizes)
        self.line_length = line_length
        self.min_length = min_length
        self.leeway = line_length - min_length

    def __getitem__(self, i):
        return self.block_tuples[i]

    def __iter__(self):
        return iter(self.block_tuples)

    def __len__(self):
        return len(self.block_sizes)

    def __str__(self):
        if self.n_unique_colors == 1:
            return str(self.block_sizes)
        else:
            return str(self.block_tuples)

    def __repr__(self):
        return f"""BlockClues object. block_sizes: {self.block_sizes}, line_length: {self.line_length}, block_colors:
        {self.block_colors}"""

    def is_empty(self) -> bool:
        """Returns True if the LineClues object has no blocks and False otherwise.

        Returns:
            A boolean."""
        return not bool(self.block_sizes)

    def get_block_size(self, i: int) -> int:
        """Returns the size of the i-th block.

        Args:
            i (int): index of the block which size is returned.

        Returns:
            An integer equal to the size of the i-th block."""
        return self.block_sizes[i]

    def get_block_tuple(self, i: int) -> tuple[int, int]:
        """Returns the 2-tuple of the size and color of the i-th block.

        Args:
            i (int): index of the block which size and (integer representation of the) color are returned.

        Returns:
            A 2-tuple of integers containing the  size and (integer representation of the) color of the i-th block."""

        return self.block_tuples[i]

    def get_block_color(self, i: int) -> str:
        """Returns the (integer representation of the) color of the i-th block.

        Args:
            i (int): index of the block which (integer representation of the) color is returned.

        Returns:
            An integer representation of the color of the i-th block."""
        return self.block_colors[i]


class Nonogram:
    """Class that represents the nonogram puzzle."""

    def __init__(self, all_line_clues: tuple[tuple[LineClues, ...], tuple[LineClues, ...]]) -> None:
        """Initializes the nonogram class object, given all row clues and column clues.

        The row clues in the tuple are ordered from top to bottom, the column clues in the tuple are ordered from left
        to right.

        Args:
            all_line_clues: tuple[tuple[LineClues, ...], tuple[LineClues, ...]]: 2-tuple with respectively a tuple of
            all row clues and a tuple of all column clues of the nonogram

        Raises:
            TypeError: if the tuples in all_line_clues are not tuples of LineClues objects.
            ValueError: if a LineClues object in all_line_clues has a minimal length larger then number of columns
        """

        if not all(isinstance(i, LineClues) for i in all_line_clues[0] + all_line_clues[1]):
            msg = "Arguments must be tuples of LineClues classes"
            logging.error(msg)
            raise TypeError(msg)

        self.all_line_clues = all_line_clues
        self.all_row_clues = all_line_clues[0]
        self.all_col_clues = all_line_clues[1]
        self.n_row = len(self.all_row_clues)
        self.n_col = len(self.all_col_clues)
        self.shape = (self.n_row, self.n_col)

        for row_or_col, all_rowcol_clues in enumerate(self.all_line_clues):
            for line_clues in all_rowcol_clues:
                if line_clues.min_length > self.shape[row_or_col - 1]:
                    msg = f"LineClues: {line_clues} too large for puzzle dimensions: {self.shape}"
                    logging.error(msg)
                    raise ValueError(msg)

    def __str__(self):
        return f"Row clues: {str(self.row_clues_tuple)}.\nColumn clues: {str(self.col_clues_tuple)}"

    def __repr__(self):
        return f"Row clues: {str(self.row_clues_tuple)}.\nColumn clues: {str(self.col_clues_tuple)}"

    def get_row_clues(self, i: int) -> LineClues:
        """Returns the LineClues object representing the i-th row of the nonogram puzzle.

        Args:
            i (int): index of the row which LineClues are returned.

        Returns:
            The LineClues object representing the i-th row of the nonogram puzzle."""
        return self.all_row_clues[i]

    def get_col_clues(self, i: int) -> LineClues:
        """Returns the LineClues object representing the i-th column of the nonogram puzzle.

        Args:
            i (int): index of the column which LineClues are returned.

        Returns:
            The LineClues object representing the i-th column of the nonogram puzzle."""
        return self.all_col_clues[i]

    def get_line_clues(self, row_or_col: int, i: int):
        """Returns the LineClues object representing the i-th row/column of the nonogram puzzle.

        Args:
            row_or_col (int): integer that determines if method is applied to row (0) or column (1)
            i (int): index of the row/column of which LineClues are returned.

        Returns:
            The LineClues object representing the i-th column of the nonogram puzzle."""
        return self.all_line_clues[row_or_col][i]


class NonogramGrid():
    """Class that represents a grid of squares of a nongram puzzle.

    The grid consists of values -1; representing unfilled/undetermined squares/values, values 0; representing empty
    squares (sometimes crossed out in a real nonogram solution), and integer values >0, representing colored squares.
    The grid can be completely filled in (only integers >= 0), but also not complete (also containing integers -1). The
    grid can be mutated."""

    def __init__(self, grid_array: np.ndarray) -> None:
        """Initializes the class object, given the grid array.

        Args:
            grid_array (np.ndarray): array with integer values >= -1, representing the nonogram grid.

        Raises:
            ValueError: if grid_array contains non-integer values.
            ValueError: if grid_array contains integers <-1.
        """

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

    def __str__(self):
        return str(self.grid_array)

    def __repr__(self):
        return str(f"NonogramGrid: {self.grid_array}")

    def __copy__(self):
        return NonogramGrid(grid_array=self.grid_array.copy())

    def set_row(self, i: int, row_array: np.ndarray) -> None:
        """Sets the i-th row of the nonogram grid to the given row.

        Args:
            i (int): index of the nonogram grid row which is set to the given row.
            row_array (np.ndarray): row to which the nonogram grid row is set."""
        self.grid_array[i, :] = row_array

    def set_col(self, i: int, col_array: np.ndarray) -> None:
        """Sets the i-th column of the nonogram grid to the given column.

        Args:
            i (int): index of the nonogram grid column which is set to the given column.
            col_array (np.ndarray): column to which the nonogram grid column is set."""
        self.grid_array[:, i] = col_array

    def set_line(self, row_or_col: int, i: int, line_array: np.ndarray) -> None:
        """Sets the i-th line (row or column) of the nonogram grid to the given line.

        Args:
            row_or_col (int): integer that determines if method is applied to row (0) or column (1)
            i (int): index of the nonogram grid line which is set to the given line.
            line_array (np.ndarray): line to which the nonogram grid line is set."""
        if row_or_col == 0:
            self.set_row(i=i, row_array=line_array)
        elif row_or_col == 1:
            self.set_col(i=i, col_array=line_array)

    def set_value(self, row_col_index: tuple[int, int], value: int) -> None:
        """Sets the index-specified element of the nonogram grid to the given value.

        Args:
            row_col_index (tuple[int, int]): index of the nonogram grid element which is set to the given value.
            value (int): value to which the nonogram grid element is set."""
        self.grid_array[row_col_index] = value

    def get_row(self, i: int) -> np.ndarray:
        """Return i-th row of the nonogram grid.

        Args:
            i (int): index of the row which is returned.

        Returns:
            The numpy array that is the i-th row of the nonogram grid."""
        return self.grid_array[i, :]

    def get_col(self, i: int) -> np.ndarray:
        """Returns i-th column of the nonogram grid.

        Args:
            i (int): index of the column which is returned.

        Returns:
            The numpy array that is the i-th column of the nonogram grid."""
        return self.grid_array[:, i]

    def get_line(self, row_or_col: int, i: int) -> np.ndarray:
        """Returns i-th line (row or column) of the nonogram grid.

        Args:
            row_or_col (int): integer that determines if method is applied to row (0) or column (1)
            i (int): index of the line which is returned.

        Returns:
            The numpy array that is the i-th line of the nonogram grid."""
        if row_or_col == 0:
            return self.get_row(i=i)
        elif row_or_col == 1:
            return self.get_col(i=i)

    def get_value(self, row_col_index: tuple[int, int]) -> int:
        """Returns the value of the index-specified element of the nonogram grid.

        Args:
            row_col_index (tuple[int, int]): index of the nonogram grid element value which is returned.

        Returns:
            The integer value of the index-specified element of the nonogram grid."""
        return self.grid_array[row_col_index]

    @classmethod
    def instantiate_empty_grid(cls, n_row: int, n_col: int) -> 'NonogramGrid':
        """Returns an instance of the nonogram grid class, with an empty grid array (containing only values -1).

        Args:
            n_row (int): number of rows of the returned nonogram grid instance.
            n_col (int): number of columns of the returned nonogram grid instance.

        Returns:
            An instance of the nonogram grid class with an empty grid array"""
        grid_array = np.full((n_row, n_col), -1)

        return cls(grid_array=grid_array)


class LinePriorityQueue():
    """Class that assigns lines to a priority queue.

    The class prevents having duplicate line in the queue, by using an set to keep track of the lines in the queue. When
    pushing lines to the queue, duplicates are replaced if they have higher priority. Priorities are tracked in separate
    priority tracker to accommodate this."""

    def __init__(self, n_row: int, n_col: int) -> None:
        """Initialize empty line priority queue.

        Nonogram dimensions are needed to build the priority tracker.

        Args:
            n_row (int): number of rows of the nonogram, for which line priority queue is used.
            n_col (int): number of columns of the nonogram, for which line priority queue is used.
        """
        self.priority_queue = []
        self.queue_set = set()
        self.priority_tracker = (n_row * [0,], n_col * [0,])

    def push(self, item: tuple[float, tuple[int, int]]) -> None:
        """Push item to line priority queue.

        Args:
            item (tuple[float, tuple[int, int]]): contains the priority of the line-tuple, and the line-tuple itself.
                                                  the line-tuple consists of an integer 0 or 1 representing whether the
                                                  line is a row or column respectively, and an index of the line.
        """
        priority, (row_or_col, line_index) = item

        if (row_or_col, line_index) in self.queue_set:
            current_priority = self.priority_tracker[row_or_col][line_index]

            if priority < current_priority:
                self.priority_queue.remove((current_priority, (row_or_col, line_index)))
                heapq.heapify(self.priority_queue)
                heapq.heappush(self.priority_queue, item)
                self.priority_tracker[row_or_col][line_index] = priority

        else:
            heapq.heappush(self.priority_queue, item)
            self.queue_set.add((row_or_col, line_index))
            self.priority_tracker[row_or_col][line_index] = priority

    def pop(self) -> tuple[float, tuple[int, int]]:
        """Returns and removes the item with the highest priority (lowest priority value) from the queue.

        Returns:
            A 2-tuple with a float (priority) and another 2-tuple representing the line of the nonogram (First item: an
            integer 0 or 1 representing whether the line is a row or column respectively. Second item: an integer
            representing index of the line itself. """
        try:
            item = heapq.heappop(self.priority_queue)
            (row_or_col, line_index) = item[1]
            self.queue_set.remove((row_or_col, line_index))
            self.priority_tracker[row_or_col][line_index] = 0
            return (row_or_col, line_index)

        # If queue is empty, return None
        except IndexError:
            return None

    def is_empty(self):
        """Returns True if queue is empty, False otherwise.

        Returns:
            A boolean, True if queue is empty, False otherwise."""
        return not bool(self.priority_queue)
