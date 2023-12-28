# We could have used "unittest" library for this, but this seemed as a more suitable way
import time
from typing import List

from constants import MINIMAL_BOARD
from helpers import get_partial_board, print_board
from solver import cost, solve_miracle_sudoku

# Configuration that all tests use
MAX_ITERATIONS = 1000
FILL_METHOD = "3x3_first_fit"
NEIGHBOURHOOD_METHOD = "only_in_squares"
EXPLORATION_EXPLOITATION_COEFFICIENT = 0.25


def test_wrapper(
    name: str,
    starting_board: List[List[int]]
):
    solving_started = time.time()
    final_board = solve_miracle_sudoku(
        starting_board=starting_board,
        max_iterations=MAX_ITERATIONS,
        fill_method=FILL_METHOD,
        neighbourhood_method=NEIGHBOURHOOD_METHOD,
        exploration_exploitation_coefficient=EXPLORATION_EXPLOITATION_COEFFICIENT,
        verbose=0
    )
    solving_finished = time.time()

    print(f"########## Test '{name}' ##########")
    print(f"$$$$$ Configuration:")
    print("Starting board:")
    print_board(starting_board)
    print(f"$$$$$ Results:")
    if final_board:
        final_cost = cost(final_board)
        print("Final board:")
        print_board(final_board)
        print(f"Final cost: {final_cost}")
        print(f"Success: {final_cost == 0}")
    else:
        print("ERROR: The board wasn't correct!")
        print("Success: False")
    print(f"Execution time: {(solving_finished - solving_started):.5f} seconds")
    print()


def test_article():
    # This is the starting board for the introduction of the miracle sudoku problem in
    #   the article
    starting_board = MINIMAL_BOARD
    test_wrapper(test_article.__name__, starting_board)


# Tests below are in different levels of completion of the full board from the miracle
#   sudoku article. Each of them has a number representing amount of spaces filled (from
#   81). Tests with 0 and 1 filled spaces have multiple possible solutions.


def test_0_empty():
    # This test has a lot of solutions, let's see if our algorithms can generate a
    #   correct solution
    starting_board = get_partial_board(filled_spaces=0)
    test_wrapper(test_0_empty.__name__, starting_board)


def test_1_partial():
    # This test has multiple solutions, let's see how our algorithm does
    starting_board = get_partial_board(filled_spaces=1)
    test_wrapper(test_1_partial.__name__, starting_board)


def test_11_partial():
    starting_board = get_partial_board(filled_spaces=11)
    test_wrapper(test_11_partial.__name__, starting_board)


def test_21_partial():
    starting_board = get_partial_board(filled_spaces=21)
    test_wrapper(test_21_partial.__name__, starting_board)


def test_41_partial():
    starting_board = get_partial_board(filled_spaces=41)
    test_wrapper(test_41_partial.__name__, starting_board)


def test_61_partial():
    starting_board = get_partial_board(filled_spaces=61)
    test_wrapper(test_61_partial.__name__, starting_board)


def test_81_full():
    # This test has a starting board that is already complete
    starting_board = get_partial_board(filled_spaces=81)
    test_wrapper(test_81_full.__name__, starting_board)


def test_impossible():
    # This test contains an illegal placing of the ones (chess horse move away)
    starting_board = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    test_wrapper(test_impossible.__name__, starting_board)


def test_wrong():
    # This test contains illegal numbers
    starting_board = [
        [0, 0, 0, 0, 0, 0, 0, 0, 333],
        [0, 0, 0, 0, 0, 0, 0, 0, -2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    test_wrapper(test_wrong.__name__, starting_board)


def main():
    for test_f in [
        test_article,
        test_0_empty,
        test_1_partial,
        test_11_partial,
        test_21_partial,
        test_41_partial,
        test_61_partial,
        test_81_full,
        test_impossible,
        test_wrong,
    ]:
        test_f()


if __name__ == "__main__":  # If the file is run directly
    main()
