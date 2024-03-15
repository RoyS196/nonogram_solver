import numpy as np
import pytest
import os
from copy import copy
from pathlib import Path

import src.helpers as helpers
import src.solving as solving
from src.nonogram import Nonogram, NonogramGrid, LineClues


class TestClass:
    # TODO: edge cases
    # TODO: multiple colors testing
    # TODO: test errors?
    # TODO: seperate file for test input?

    # Test parameters for array compatibility check and array update
    line_check_empty = np.full(shape=(5,), fill_value=-1)
    line_check_non_empty = np.array(object=[-1, 0, -1, -1, 1])
    line_check_compatible = np.array(object=[1, 0, 0, -1, -1])
    line_check_not_compatible = np.array(object=[0, 1, -1, 1, -1])
    line_check_compatible_result = np.array(object=[1, 0, 0, -1, 1])

    @pytest.mark.parametrize("potential_line, current_line, result",
                             [(line_check_empty, line_check_empty, True),
                              (line_check_non_empty, line_check_empty, True),
                              (line_check_compatible, line_check_non_empty, True),
                              (line_check_not_compatible, line_check_non_empty, False),
                              ])
    def test_check_arrays_compatibility(self, potential_line, current_line, result):

        assert solving.check_arrays_compatibility(array_1=potential_line, array_2=current_line) == result

    @pytest.mark.parametrize("new_line, current_line, result",
                             [(line_check_empty, line_check_empty, line_check_empty),
                              (line_check_non_empty, line_check_empty, line_check_non_empty),
                              (line_check_compatible, line_check_non_empty, line_check_compatible_result),
                              ])
    def test_update_array_values(self, new_line, current_line, result) -> None:
        current_line_copy = current_line.copy()

        solving.update_array_values(base_array=current_line_copy, array_update=new_line)

        np.testing.assert_array_equal(current_line_copy, result)

    def test_obtain_arrays_intersection(self) -> None:
        list_lines = [np.array(object=[1, 1, 0, 0, 0, 2, 3]),
                      np.array(object=[1, -1, 1, 0, -1, 2, 3]),
                      np.array(object=[1, 0, 0, 0, 0, 2, -1])]
        result = np.array(object=[1, -1, -1, 0, -1, 2, -1])

        np.testing.assert_equal(solving.obtain_arrays_intersection(list_arrays=list_lines), result)

    # Test parameters for line leeway solving method
    line_array_empty = np.full(shape=(10,), fill_value=-1)  # TODO: non-empty line
    line_clues_zeros = LineClues(block_sizes=(), line_length=10)
    result_zeros = np.full(shape=(10,), fill_value=0)
    line_clues_ones = LineClues(block_sizes=(10,), line_length=10)
    result_ones = np.full(shape=(10,), fill_value=1)
    line_clues_full = LineClues(block_sizes=(1, 2, 5), line_length=10)
    result_full = np.array(object=[1, 0, 1, 1, 0, 1, 1, 1, 1, 1])
    line_clues_overlap = LineClues(block_sizes=(5, 2), line_length=10)
    result_overlap = np.array(object=[-1, -1, 1, 1, 1, -1, -1, -1, -1, -1])
    line_clues_no_overlap = LineClues(block_sizes=(2, 1), line_length=10)
    result_no_overlap = np.full(shape=(10,), fill_value=-1)
    line_clues_colored = LineClues(block_sizes=(4, 1, 3), line_length=10, block_colors=[1, 2, 3])
    result_colored = np.array(object=[-1, -1, 1, 1, -1, -1, -1, 3, -1, -1])
    line_clues_colored_repeating = LineClues(block_sizes=(4, 1, 3), line_length=10, block_colors=[1, 1, 2])
    result_colored_repeating = np.array(object=[-1, 1, 1, 1, -1, -1, -1, 2, 2, -1])

    @pytest.mark.parametrize("line_clues, line_array, result",
                             [(line_clues_zeros, line_array_empty, result_zeros),
                              (line_clues_ones, line_array_empty, result_ones),
                              (line_clues_full, line_array_empty, result_full),
                              (line_clues_overlap, line_array_empty, result_overlap),
                              (line_clues_no_overlap, line_array_empty, result_no_overlap),
                              (line_clues_colored, line_array_empty, result_colored),
                              (line_clues_colored_repeating, line_array_empty, result_colored_repeating),
                              ])
    def test_line_leeway_solving_step(self, line_clues: LineClues, line_array: np.ndarray, result: np.ndarray):
        line_array_copy = line_array.copy()

        solving.line_leeway_solving_step(line_clues=line_clues, line_array=line_array_copy)

        np.testing.assert_array_equal(line_array_copy, result)

    def test_grid_leeway_solving_method(self):
        project_folder = Path(os.path.realpath(__file__)).parent.parent

        # TODO: hardcode test grids, more cases, parametrize
        grid_test = NonogramGrid.instantiate_empty_grid(n_row=15, n_col=10)
        grid_complete = helpers.obtain_grid_from_csv(filename=f"{project_folder}/tests/test_files/grid_monalisa.csv")
        grid_leeway = helpers.obtain_grid_from_csv(filename=f"{project_folder}/tests/test_files/grid_monalisa_leeway.csv")
        nonogram = helpers.obtain_nonogram_from_grid(nonogram_grid=grid_complete)

        solving.grid_leeway_solving_step(nonogram=nonogram, current_grid=grid_test)

        np.testing.assert_array_equal(grid_test.grid_array, grid_leeway.grid_array)

    # Test parameters for determining line possibilities
    line_clues = LineClues(block_sizes=(2, 2, 3), line_length=10, block_colors=(1, 1, 2))
    result = {(1, 1, 0, 1, 1, 2, 2, 2, 0, 0),
              (1, 1, 0, 1, 1, 0, 2, 2, 2, 0),
              (1, 1, 0, 0, 1, 1, 2, 2, 2, 0),
              (0, 1, 1, 0, 1, 1, 2, 2, 2, 0),
              (1, 1, 0, 1, 1, 0, 0, 2, 2, 2),
              (1, 1, 0, 0, 1, 1, 0, 2, 2, 2),
              (0, 1, 1, 0, 1, 1, 0, 2, 2, 2),
              (1, 1, 0, 0, 0, 1, 1, 2, 2, 2),
              (0, 1, 1, 0, 0, 1, 1, 2, 2, 2),
              (0, 0, 1, 1, 0, 1, 1, 2, 2, 2)}
    base_line = np.array(object=[1, -1, -1, -1, -1, -1, -1, -1, -1, 0])
    result_base_line = {(1, 1, 0, 1, 1, 2, 2, 2, 0, 0),
                        (1, 1, 0, 1, 1, 0, 2, 2, 2, 0),
                        (1, 1, 0, 0, 1, 1, 2, 2, 2, 0)}
    base_line_impossible = np.array(object=[2, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    result_empty_set = set()

    @pytest.mark.parametrize("line_clues, base_line, result",
                             [(line_clues, None, result),
                              (line_clues, base_line, result_base_line),
                              (line_clues, base_line_impossible, result_empty_set),
                              ])
    def test_determine_possible_lines(self, line_clues, base_line, result):

        assert solving.determine_possible_lines(line_clues=line_clues, base_line=base_line) == result
