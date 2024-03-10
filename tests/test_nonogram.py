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

    project_folder = Path(os.path.realpath(__file__)).parent.parent

    # Normal test cases for line check and line update
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
    def test_check_potential_line(self, potential_line, current_line, result):

        assert solving.check_potential_line(potential_line=potential_line, current_line=current_line) == result

    @pytest.mark.parametrize("new_line, current_line, result",
                             [(line_check_empty, line_check_empty, line_check_empty),
                              (line_check_non_empty, line_check_empty, line_check_non_empty),
                              (line_check_compatible, line_check_non_empty, line_check_compatible_result),
                              ])
    def test_update_current_line(self, new_line, current_line, result) -> None:
        current_line_copy = current_line.copy()

        solving.update_current_line(new_line=new_line, current_line=current_line_copy)

        np.testing.assert_array_equal(current_line_copy, result)

    def test_obtain_lines_intersection(self) -> None:
        list_lines = [np.array(object=[1, 1, 0, 0, 0, 2, 3]),
                      np.array(object=[1, -1, 1, 0, -1, 2, 3]),
                      np.array(object=[1, 0, 0, 0, 0, 2, -1])]
        result = np.array(object=[1, -1, -1, 0, -1, 2, -1])

        np.testing.assert_equal(solving.obtain_lines_intersection(list_lines=list_lines), result)

    # Normal test cases for line leeway
    current_line_empty = np.full(shape=(10,), fill_value=-1)  # TODO: non-empty line
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

    @pytest.mark.parametrize("line_clues, current_line, result",
                             [(line_clues_zeros, current_line_empty, result_zeros),
                              (line_clues_ones, current_line_empty, result_ones),
                              (line_clues_full, current_line_empty, result_full),
                              (line_clues_overlap, current_line_empty, result_overlap),
                              (line_clues_no_overlap, current_line_empty, result_no_overlap),
                              (line_clues_colored, current_line_empty, result_colored),
                              (line_clues_colored_repeating, current_line_empty, result_colored_repeating)
                              ])
    def test_line_leeway_method(self, line_clues, current_line, result):
        current_line_copy = current_line.copy()

        solving.line_leeway_method(line_clues=line_clues, current_line=current_line_copy)

        np.testing.assert_array_equal(current_line_copy, result)

    # TODO: hardcode example grids, or test helper functions
    grid_empty = NonogramGrid.instantiate_empty_grid(n_row=15, n_col=10)
    grid_monalisa = helpers.obtain_grid_from_csv(
        filename=f"{project_folder}/tests/test_files/grid_monalisa.csv",
        delimiter=",")
    grid_monalisa_leeway = helpers.obtain_grid_from_csv(
        filename=f"{project_folder}/tests/test_files/grid_monalisa_leeway.csv",
        delimiter=",")
    nonogram_monalisa = helpers.obtain_nonogram_from_grid(nonogram_grid=grid_monalisa)

    @pytest.mark.parametrize("nonogram, current_grid, result",
                             [(nonogram_monalisa, grid_empty, grid_monalisa_leeway),
                              ])
    def test_grid_leeway_method(self, nonogram: Nonogram, current_grid: NonogramGrid, result: NonogramGrid):
        current_grid_copy = copy(current_grid)

        solving.grid_leeway_method(nonogram=nonogram, current_grid=current_grid_copy)

        np.testing.assert_array_equal(current_grid_copy.grid_array, result.grid_array)
