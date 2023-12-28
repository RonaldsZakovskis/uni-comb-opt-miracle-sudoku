from constants import MINIMAL_BOARD
from helpers import get_partial_board, print_board
from solver import cost, solve_miracle_sudoku


def main():
    # $$$$$ Configuration $$$$$
    max_iterations = 10000
    # max_iterations = None  # Runs until fully solved

    # 1 – print necessary stuff | 2 – print everything (useful for debugging)
    verbose = 1

    # (1) All the unused numbers are entered one-by-one randomly
    # fill_method = "full_random"
    # (2) All the unused numbers in every 3x3 are entered randomly one-by-one (after
    #   this, each of the 3x3 blocks definitely have each number 1-9)
    # fill_method = "3x3_random"
    # (3) Go each position by each position, go through all numbers that still don't
    #   have 9 copies of, and enter the one that has the lowest cost
    # fill_method = "full_first_fit"
    # (4) Go through each of the 3x3s, go through each of the missing spots one-by-one,
    #   enter the number with the lowest cost
    fill_method = "3x3_first_fit"

    # (1) Looks at all possible combinations of the empty spots for moves
    # neighbourhood_method = "all_moves"
    # (2) Looks at all possible combinations of two empty spots in the 3x3 squares.
    #   Note: Only makes sense if the board has one of each number in each 3x3, that is,
    #   3x3_random or 3x3_first_fit is chosen.
    neighbourhood_method = "only_in_squares"

    # Should we explore or exploit, or toggle?
    # 0 == 100 % Random moves
    # 0.75 == 25 % Random moves and 75 % Best cost moves
    # 1 == 100 % Best cost moves
    exploration_exploitation_coefficient = 0.25

    # SETTING THE BOARD
    enter_board_manually = False
    if enter_board_manually:  # If you want to enter the board manually
        starting_board = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    else:  # If you want the board to be generated automatically
        use_partial_board = True
        if not use_partial_board:  # (1) Article board (starting version)
            starting_board = MINIMAL_BOARD
        else:  # (2) Article board (filled version) with space dropping applied
            filled_spaces = 33  # Value needs to be between 0 and 81
            starting_board = get_partial_board(filled_spaces=filled_spaces)

    # $$$$$ Solving the miracle sudoku board $$$$$
    final_board = solve_miracle_sudoku(
        starting_board=starting_board,
        max_iterations=max_iterations,
        fill_method=fill_method,
        neighbourhood_method=neighbourhood_method,
        exploration_exploitation_coefficient=exploration_exploitation_coefficient,
        verbose=verbose
    )

    # $$$$$ Analysing the results $$$$$
    if final_board:
        print("Final board:")
        print_board(final_board)
        print(f"Final cost: {cost(final_board)}")


if __name__ == "__main__":  # If the file is run directly
    main()
