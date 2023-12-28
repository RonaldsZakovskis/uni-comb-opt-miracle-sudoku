import copy
import random

from constants import FULL_BOARD
from typing import Any, List


def print_board(board: List[List[int]], verbose: bool = True) -> None:
    """ Prints the passed board if the verbosity is set high enough.

    If the board has missing values, they will be printed as "_".

    Args:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board. It can contain empty values.
        verbose: If true, passed board is printed.

    Returns:
        None.
    """
    if verbose:
        for line_index, line in enumerate(board):
            for digit_index, digit in enumerate(line):
                print(get_board_symbol(digit), end=" ")
                if digit_index % 3 == 2 and digit_index != 8:
                    print("|", end=" ")
            print()
            if line_index % 3 == 2 and line_index != 8:
                print("-" * 21)


def get_board_symbol(digit: int) -> str:
    """ Returns the symbol representation for the specific digit.

    If the board has missing values, they will be printed as "_".

    Args:
        digit: Value of the specific cell in the sudoku board. We assume it's between 0
            and 9.

    Returns:
        symbol: Symbol to print out as the value in the sudoku board.
    """
    return "_" if digit == 0 else str(digit)


def verbose_print(printable: Any, verbose: bool = True) -> None:
    """ Prints the passed value if the verbosity is set high enough.

    Args:
        printable: Stuff to print out.
        verbose: If true, passed value is printed.

    Returns:
        None.
    """
    if verbose:
        print(printable)


def get_partial_board(filled_spaces: int) -> List[List[int]]:
    """ Returns a partially filled miracle sudoku board. Especially useful for testing.

    We take the fully solved board from the article and replace random spots with empty
        ones.

    Args:
        filled_spaces: Amount of spaces that are filled in the miracle sudoku board.

    Returns:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board. It has filled_spaces filled spaces and 81 - filled_spaces
            empty spaces.
    """
    missing_spaces = 81 - filled_spaces

    if missing_spaces == 0:
        return FULL_BOARD
    else:
        # Will hold all possible positions: [0, 0], [0, 1], ..., [8, 8]
        possible_positions = []
        for i in range(9):
            for j in range(9):
                possible_positions.append([i, j])

        # Shuffle the positions
        random.shuffle(possible_positions)

        # First n (missing_spaces) random positions will be emptied
        empty_positions = possible_positions[:missing_spaces]

        # Copying the full board
        partial_board = copy.deepcopy(FULL_BOARD)

        for empty_position in empty_positions:
            i, j = empty_position
            partial_board[i][j] = 0

        return partial_board
