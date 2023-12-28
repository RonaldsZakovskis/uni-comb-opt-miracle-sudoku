import copy
from itertools import combinations
import math
import random
from typing import List, Optional, Tuple

from helpers import print_board, verbose_print


def solve_miracle_sudoku(
    starting_board: List[List[int]],
    max_iterations: int = None,
    fill_method: str = "3x3_first_fit",
    neighbourhood_method: str = "all_moves",
    exploration_exploitation_coefficient: float = 0,
    verbose: int = 1
) -> Optional[List[List[int]]]:
    """ Tries to complete the given miracle sudoku board using optimization.

    If the board is incorrect, complete or only missing one spot, no optimization is
        run, otherwise it is, in that case, depending on the configuration and luck, the
        board is returned partially or fully completed.

    Args:
        starting_board: Two dimensional list of integers describing the starting state
            of the miracle sudoku board.
        max_iterations: Number of iterations we should run at maximum until reaching
            cost of 0 or running out of iterations. If not provided, it won't stop
            unless the cost drops to 0, that is, the board is solved.
        fill_method: Method describing how to fill the empty spaces in the starting
            board. Must be one of "full_random", "3x3_random", "full_first_fit" or
            "3x3_first_fit". "full_random" – all the unused numbers are entered
            one-by-one randomly. "3x3_random" – all the unused numbers in every 3x3 are
            entered randomly one-by-one (after this, each of the 3x3 blocks definitely
            have each number from 1 to 9. "full_first_fit" – go each position by each
            position, go through all numbers that still don't have 9 copies of, and
            enter the one that has the lowest cost. "3x3_first_fit – go through each of
            the 3x3s, go through each of the missing spots one-by-one, enter the number
            that results with the lowest cost.
        neighbourhood_method: Method describing how to get the neighborhood (moves).
            Must be "all_moves" or "only_in_squares". "all_moves" – all possible
            combinations of the empty spaces are considered as all possible moves.
            "only_in_squares" – moves are only all empty spaces swaps in every one of
            the 3x3s, this only makes sense if the board is guaranteed to have every
            number from 1 to 9 after the initial filling of the board.
        exploration_exploitation_coefficient: Coefficient describing if we should
            explore, exploit, or toggle. 0 == 100 % Random moves. 0.75 == 25 % Random
            moves and 75 % Best cost moves. 1 == 100 % Best cost moves.
        verbose: Level of detail to output to the console. Must be 0, 1 or 2. 0 means
            nothing is printed, 1 means just the most necessary stuff is printed, 2
            means everything is printed.

    Returns:
        final_board: Two dimensional list describing the final state of the miracle
            sudoku board. It has no missing values, the board in total has the correct
            amount of each number, but it may or may not be solved, that is, the board's
            cost may or may not be 0. If the board was incorrect, None is returned.
    """
    # Look at the starting board
    if len(starting_board) != 9:
        verbose_print(
            "ERROR: Incorrect amount of rows in the starting board!",
            verbose > 0
        )
        return None
    for row in starting_board:
        if len(row) != 9:
            verbose_print(
                "ERROR: Incorrect amount of full columns in the starting board!",
                verbose > 0
            )
            return None
    verbose_print("Starting board:", verbose > 0)
    print_board(starting_board, verbose > 0)
    starting_board_cost = cost(starting_board, verbose == 2)
    if starting_board_cost > 0:
        verbose_print("ERROR: Starting board contains rule breaking!", verbose > 0)
        return None
    else:
        verbose_print("Starting board doesn't break any of the rules!", verbose > 0)

    starting_board_number_counts = count_numbers(starting_board)
    if sum(starting_board_number_counts) != 81:
        verbose_print(
            "ERROR: Starting board contains integers that aren't between 0 and 9!",
            verbose > 0
        )
        return None
    missing_spaces = starting_board_number_counts[0]
    verbose_print(f"Missing spaces: {missing_spaces}\n", verbose > 0)

    if missing_spaces == 0:
        verbose_print("The board is already complete!", verbose > 0)
        return starting_board

    # Construct the initial solution – fill empty places (0s) with some values
    initial_board = fill_board(starting_board, fill_method, verbose == 2)
    verbose_print("Initially filled board:", verbose > 0)
    print_board(initial_board, verbose > 0)
    initial_cost = cost(initial_board, verbose == 2)
    verbose_print(f"Initially filled board cost: {initial_cost}\n", verbose > 0)
    # If there was only one value missing or pure luck, the board is already correctly
    #   filled
    if initial_cost == 0:
        verbose_print(
            "The board was filled using just filling the values somewhat smartly",
            verbose > 0
        )
        return initial_board

    # We could add some extra function that does easy number placings, for example, a
    #   row has everything except 1 number, in that case, we can set that instantly to
    #   the missing value, to make the optimizer's job easier

    # $$$$$ Configuration for Simulated Annealing $$$$$
    # (1) Let's set the temperature as the square root of missing spaces, that would
    #   result in initial temperature between 9 (sqrt81) and 1 (sqrt 1), this is a nice
    #   way, because it scales on the amount of missing spaces
    initial_temperature = math.sqrt(missing_spaces)
    # (2) We could set it to a constant
    # initial_temperature = 3
    # (3) Better option might be to generate many versions of the initial board and
    #   measure cost to decide the temperature, then use standard deviation or some
    #   other metric to decide the temperature

    # Temperature multiplier for decreasing, must be between 0 and 1 excluding both
    #   0.99  – slow and steady
    #   0.95 or even 0.5 – more aggressive
    cooling_rate = 0.95

    # We could have had a different number of iterations for each of the temperatures,
    #   but I chose to have the same amount for all of them
    # A lot of random moves will probably be bad, so this number should be high, so that
    #   the temperature doesn't decrease before a single decent move is randomly found,
    #   but it would be cool for it to be dependent on the size of the board, so let's
    #   use missing spaces
    iterations_per_t = 5 * missing_spaces

    temperature = initial_temperature
    current_best_cost = initial_cost
    current_best_board = initial_board
    current_cost = initial_cost
    current_board = initial_board

    # Try to minimize cost from, for example, 275 to 0
    iterations = 1
    while True:
        verbose_print(f"Iteration Nr. {iterations}", verbose > 0)
        verbose_print(f"Best cost: {current_best_cost}", verbose > 0)
        verbose_print(f"Current cost: {current_cost}", verbose > 0)

        # Get neighborhood – possible 2 position swaps, non-empty
        # We could have also kept a Tabu list of recent moves to not repeat them, that
        #   might have been useful
        # This function maybe should have been made into a generator function, to make
        #   it faster, including the functions below
        neighborhood = get_neighborhood(
            starting_board,
            neighbourhood_method,
            verbose == 2
        )

        # Compare neighborhood
        # For each 2 positions, create a new board by swapping the two positions...
        # This is quite slow, probably creating a generator could go a long way
        neighborhood_scores = compare_neighborhood(
            current_board,
            neighborhood,
            verbose == 2
        )

        # Choose a move from the neighborhood
        # Simulated Annealing or Hill Climbing alone didn't seem to do so well, so I had
        #   an Exploration-exploitation dilemma
        # Choose only random moves in the neighborhood
        if exploration_exploitation_coefficient == 0:
            explore_or_exploit = 0
        # Choose only the best (in cost) moves in the neighborhood
        elif exploration_exploitation_coefficient == 1:
            explore_or_exploit = 1
        # Sometimes do a random move, sometimes do the best cost move
        else:
            if random.uniform(0, 1) < exploration_exploitation_coefficient:
                explore_or_exploit = 1
            else:
                explore_or_exploit = 0

        if explore_or_exploit == 1:  # Best move
            verbose_print("Looking at the best move in the neighborhood", verbose > 0)
            best_neighborhood_score = min(neighborhood_scores)
            best_neighborhood_move = neighborhood[neighborhood_scores.index(best_neighborhood_score)]
        else:  # Random move
            verbose_print("Looking at a random move from the neighborhood", verbose > 0)
            chosen_move_id = random.randrange(len(neighborhood_scores))
            best_neighborhood_score = neighborhood_scores[chosen_move_id]
            best_neighborhood_move = neighborhood[chosen_move_id]

        verbose_print(
            f"Looking at move – swap {best_neighborhood_move[0]} and {best_neighborhood_move[1]}",
            verbose > 0
        )
        verbose_print(
            f"This move would result to cost: {best_neighborhood_score}",
            verbose > 0
        )

        new_cost = best_neighborhood_score
        cost_difference = current_cost - new_cost
        if cost_difference >= 0:
            verbose_print("Cost decreased, yes!", verbose > 0)
            # Do the move
            current_board = move_swap(current_board, best_neighborhood_move)
            current_cost = best_neighborhood_score
        else:
            verbose_print("Cost is worse!", verbose > 0)
            rolled_probability = random.uniform(0, 1)
            verbose_print(f"Rolled probability: {rolled_probability}", verbose > 0)
            acceptance_probability = math.exp(cost_difference / temperature)
            verbose_print(
                f"Acceptance probability: {acceptance_probability} using temperature {temperature}",
                verbose > 0
            )
            if rolled_probability < acceptance_probability:
                verbose_print(
                    f"Dice says do the move! ({rolled_probability} < {acceptance_probability})!",
                    verbose > 0
                )
                current_board = move_swap(current_board, best_neighborhood_move)
                current_cost = best_neighborhood_score
            else:
                verbose_print(
                    f"Dice says don't do the move! ({rolled_probability} >= {acceptance_probability})!",
                    verbose > 0
                )

        # If the solution is better than the previous best, save it
        if current_cost < current_best_cost:
            current_best_board = current_board
            current_best_cost = current_cost

        # If the time for lowering temperature has come
        if iterations % iterations_per_t == 0:
            temperature *= cooling_rate

        verbose_print(f"Board after Iteration Nr.{iterations}", verbose > 0)
        print_board(current_board, verbose > 0)

        if current_best_cost == 0 or iterations == max_iterations:
            break
        iterations += 1

    # Current best solution is the final solution
    final_board = current_best_board

    return final_board


def cost(board: List[List[int]], verbose: bool = False) -> int:
    """ Calculates and returns the cost of the given miracle sudoku board.

    It goes through all possible combinations of non-zero (non-empty) positions and
        checks for parts that have illegal placement.

    Args:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board.
        verbose: If true, details are printed.

    Returns:
        total_cost: Amount of rule breaks.
    """
    verbose_print("Evaluating cost of the board...", verbose)
    total_cost = 0

    # Will hold all possible positions: [0, 0], [0, 1], ..., [8, 8]
    possible_positions = []
    for i in range(9):
        for j in range(9):
            possible_positions.append([i, j])

    # Will hold add possible two position combinations:
    #   ([0, 0], [0, 1])
    #   ([0, 0], [0, 2])
    #   ...
    #   ([8, 7], [8, 8])
    position_combinations = list(combinations(possible_positions, 2))

    for two_positions in position_combinations:
        two_position_cost = 0
        first_row, first_column = two_positions[0]
        second_row, second_column = two_positions[1]
        first_value = board[first_row][first_column]
        second_value = board[second_row][second_column]

        # If one of the positions holds a missing value, we will ignore this pair
        if first_value == 0 or second_value == 0:
            continue

        # Same value on the same row
        if first_row == second_row and first_value == second_value:
            verbose_print(
                f"Same value on the same row for positions {two_positions}",
                verbose
            )
            two_position_cost += 1

        # Same value on the same column
        if first_column == second_column and first_value == second_value:
            verbose_print(
                f"Same value on the same column for positions {two_positions}",
                verbose
            )
            two_position_cost += 1

        # Same value in the same 3x3
        # Checking if both positions are in the same 3x3 square
        if first_row // 3 == second_row // 3:
            if first_column // 3 == second_column // 3:
                # Check if both values are the same
                if first_value == second_value:
                    verbose_print(
                        f"Same value in the same 3x3 for positions {two_positions}",
                        verbose
                    )
                    two_position_cost += 1

        # Same value chess knight's move away
        #   ? A ? A ?
        #   B ? ? ? B
        #   ? ? X ? ?
        #   B ? ? ? B
        #   ? A ? A ?
        # Both positions are a "2 rows 1 column" (A) chess knight's move away
        if abs(first_row - second_row) == 2 and abs(first_column - second_column) == 1:
            # Check if both values are the same
            if first_value == second_value:
                verbose_print(
                    f"Same value 2 rows 1 column knight's move away for positions {two_positions}",
                    verbose
                )
                two_position_cost += 1
        # Both positions are a "1 row 2 columns" (B) chess knight's move away
        if abs(first_row - second_row) == 1 and abs(first_column - second_column) == 2:
            # Check if both values are the same
            if first_value == second_value:
                verbose_print(
                    f"Same value 1 row 2 columns knight's move away for positions {two_positions}",
                    verbose
                )
                two_position_cost += 1

        # Same value chess king's move away
        # Both positions are a king's move away
        if abs(first_row - second_row) <= 1 and abs(first_column - second_column) <= 1:
            # Check if both values are the same
            if first_value == second_value:
                verbose_print(
                    f"Same value king's move away for positions {two_positions}",
                    verbose
                )
                two_position_cost += 1

        # Consecutive values for orthogonally adjacent positions
        # Both positions are orthogonally adjacent
        #   ? A ?
        #   B 2 C
        #   ? D ?
        # A, B, C, D can’t be one, nor three.
        if abs(first_row - second_row) + abs(first_column - second_column) == 1:
            # Check if both values are consecutive
            if abs(first_value - second_value) == 1:
                verbose_print(
                    f"Consecutive values for orthogonally adjacent positions {two_positions}",
                    verbose
                )
                two_position_cost += 1

        if two_position_cost > 0:
            verbose_print(
                f"Cost gain of {two_position_cost} from {two_positions}",
                verbose
            )
            total_cost += two_position_cost

    verbose_print(f"Total cost of the board: {total_cost}", verbose)

    return total_cost


def count_numbers(board: List[List[int]]) -> List[int]:
    """ Count and return numbers on the board.

    It counts only values that are between 0 and 9, that is, if the board is filled
        correctly, the sum of the values will be 81.

    Args:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board.

    Returns:
        nbr_instances: Amount of instances for numbers from 0 to 9.
    """
    nbr_instances = [0] * 10
    for row in board:
        for cell in row:
            if 0 <= cell <= 9:
                nbr_instances[cell] += 1
    return nbr_instances


def fill_board(
    board: List[List[int]],
    mode: str,
    verbose: bool = False
) -> List[List[int]]:
    """ Fills the passed board using the specified filling method and returns the
        calculated board.

    Args:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board. We assume that it has at least 1 missing value, and it has a
            cost of 0.
        mode: Method describing how to fill the empty spaces in the starting board. Must
            be one of "full_random", "3x3_random", "full_first_fit" or "3x3_first_fit".
            "full_random" – all the unused numbers are entered one-by-one randomly.
            "3x3_random" – all the unused numbers in every 3x3 are entered randomly
            one-by-one (after this, each of the 3x3 blocks definitely have each number
            from 1 to 9). "full_first_fit" – go each position by each position, go
            through all numbers that still don't have 9 copies of, and enter the one
            that has the lowest cost. "3x3_first_fit – go through each of the 3x3s, go
            through each of the missing spots one-by-one, enter the number that results
            with the lowest cost.
        verbose: If true, details are printed.

    Returns:
        filled_board: Two dimensional list of integers describing the state of the
            miracle sudoku board. All the missing spots have been filled. The board now
            contains each of the values from 1 to 9 each 9 times.
    """
    verbose_print(
        "Constructing the initial solution by filling empty places (0s) with some values...",
        verbose
    )

    assert mode in [
        "full_random",
        "3x3_random",
        "full_first_fit",
        "3x3_first_fit"
    ], "ERROR: Unknown filling mode specified!"

    if mode == "full_random":
        verbose_print("Filling each of the empty spots randomly...", verbose)

        nbr_instances = count_numbers(board)

        nbrs_remaining = []
        for nbr, instances in enumerate(nbr_instances):
            if nbr == 0:
                continue
            nbrs_remaining += [nbr] * (9 - instances)
        # nbrs_remaining now is something like [1, 1, 1, 2, 2, 2, 2, 2, 2, ..., 8, 8, 9]
        verbose_print(f"Remaining numbers: {nbrs_remaining}", verbose)

        # Let's shuffle it, so it's in a random order
        random.shuffle(nbrs_remaining)
        # nbrs_remaining now is something like [9, 2, 8, 2, 8, 2, 1, 1, 2, ..., 1, 2, 2]
        verbose_print(f"Remaining numbers shuffled: {nbrs_remaining}", verbose)

        # Let's create a new board, so we don't change the original
        filled_board = copy_board(board)

        filled = 0
        for id_row, row in enumerate(board):
            for id_cell, cell in enumerate(row):
                if cell == 0:
                    filled_board[id_row][id_cell] = nbrs_remaining[filled]
                    filled += 1

        return filled_board
    elif mode == "3x3_random":
        verbose_print("Filling each of the empty spots in 3x3s randomly...", verbose)

        # Let's create a new board, so we don't change the original
        filled_board = copy.deepcopy(board)

        for i in range(3):  # There are three 3x3 squares horizontally
            for j in range(3):  # There are three 3x3 squares vertically
                # Complete i-th j-th 3x3
                verbose_print(f"Look at the i={i}, j={j} 3x3:", verbose)

                # Let's find out which values are present already in the 3 by 3
                present_in_3x3 = [False] * 9
                for x in range(3):
                    for y in range(3):
                        val = board[x + i * 3][y + j * 3]
                        if val != 0:
                            present_in_3x3[val - 1] = True

                # present_in_3x3 now is something like [True, False, True, ..., False],
                #   which would mean 1 and 3 was present, but 2 and 9 wasn't

                # All non-present numbers make up the remaining numbers (in the 3x3)
                nbrs_remaining = []
                for index, existence in enumerate(present_in_3x3):
                    if not existence:
                        nbrs_remaining.append(index + 1)

                # remaining_numbers now is something like [2, 7, 8]
                verbose_print(f"Remaining numbers: {nbrs_remaining}", verbose)

                if len(nbrs_remaining) == 0:  # The 3x3 is already filled
                    continue

                # Let's shuffle it, so it's in a random order
                random.shuffle(nbrs_remaining)
                # remaining_numbers now is something like [8, 2, 7]
                verbose_print(f"Remaining numbers shuffled: {nbrs_remaining}", verbose)

                filled = 0
                for x in range(3):
                    for y in range(3):
                        val = board[x + i * 3][y + j * 3]
                        if val == 0:
                            filled_board[x + i * 3][y + j * 3] = nbrs_remaining[filled]
                            filled += 1
        return filled_board
    elif mode == "full_first_fit":
        verbose_print(
            "Filling each of the empty spots by finding the digit with the best cost...",
            verbose
        )

        # Count the numbers on the board from 0 to 9, for example,
        #   [48, 4, 4, 5, 3, 4, 3, 3, 4, 3]
        nbr_instances = count_numbers(board)

        # Calculate how many 1s, 2s, ..., 9s are remaining, for example,
        #   [6, 6, 4, 7, 2, 6, 5, 5, 7], ignore the first value, as it is the 0s
        nbr_instances_remaining = [9 - i for i in nbr_instances[1:]]

        # Make it into a dictionary for easier processing
        # {
        #   1: 6,
        #   2: 6,
        #   ...
        #   9: 7
        # }
        nbr_instances_remaining = dict(enumerate(nbr_instances_remaining, start=1))

        # Let's create a new board, so we don't change the original
        filled_board = copy_board(board)

        for id_row, row in enumerate(board):
            for id_cell, cell in enumerate(row):
                if cell == 0:  # If the cell is empty
                    verbose_print(f"Filling position [{id_row}, {id_cell}]:", verbose)
                    best_number = None
                    best_cost = None
                    # Order to evaluate costs – [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    nbr_order = list(range(1, 10))
                    # Let's shuffle it, so it's in a random order
                    # Now it's – [4, 7, 6, 1, 2, 8, 5, 9, 3]
                    random.shuffle(nbr_order)

                    for number in nbr_order:
                        amount = nbr_instances_remaining[number]
                        if amount != 0:  # If there are free instances of the digit left
                            potential_board = copy.deepcopy(filled_board)
                            potential_board[id_row][id_cell] = number
                            some_cost = cost(potential_board)
                            verbose_print(
                                f"Choosing number: {number} would result to cost: {some_cost}",
                                verbose
                            )
                            if best_cost is None or some_cost < best_cost:
                                best_cost = some_cost
                                best_number = number
                    verbose_print(
                        f"Number with lowest cost: {best_number} (cost = {best_cost})",
                        verbose
                    )
                    nbr_instances_remaining[best_number] -= 1
                    filled_board[id_row][id_cell] = best_number
        return filled_board
    elif mode == "3x3_first_fit":
        verbose_print(
            "Filling each of the empty spots in 3x3s by finding the digit with the best cost...",
            verbose
        )

        # Let's create a new board, so we don't change the original
        filled_board = copy.deepcopy(board)

        for i in range(3):  # There are three 3x3 squares horizontally
            for j in range(3):  # There are three 3x3 squares vertically
                # Complete i-th j-th 3x3
                verbose_print(f"Look at the i={i}, j={j} 3x3:", verbose)

                # Let's find out which values are present already in the 3 by 3
                present_in_3x3 = [False] * 9
                for x in range(3):
                    for y in range(3):
                        val = board[x + i * 3][y + j * 3]
                        if val != 0:
                            present_in_3x3[val - 1] = True

                # present_in_3x3 now is something like [True, False, True, ..., False],
                #   which would mean 1 and 3 was present, but 2 and 9 wasn't

                # All non-present numbers make up the remaining numbers (in the 3x3)
                nbrs_remaining = []
                for index, existence in enumerate(present_in_3x3):
                    if not existence:
                        nbrs_remaining.append(index + 1)

                # remaining_numbers now is something like [2, 7, 8]
                verbose_print(f"Remaining numbers: {nbrs_remaining}", verbose)

                if len(nbrs_remaining) == 0:  # The 3x3 is already filled
                    continue

                # Let's shuffle it, so it's in a random order
                random.shuffle(nbrs_remaining)
                # remaining_numbers now is something like [8, 2, 7]
                verbose_print(f"Remaining numbers shuffled: {nbrs_remaining}", verbose)

                for x in range(3):
                    for y in range(3):
                        val = board[x + i * 3][y + j * 3]
                        if val == 0:
                            verbose_print(
                                f"Filling position [{x + i * 3}, {y + j * 3}]:",
                                verbose
                            )
                            if len(nbrs_remaining) == 1:
                                filled_board[x + i * 3][y + j * 3] = nbrs_remaining[0]
                            else:
                                # Find number with the best cost
                                best_number = None
                                best_cost = None
                                for number in nbrs_remaining:
                                    potential_board = copy.deepcopy(filled_board)
                                    potential_board[x + i * 3][y + j * 3] = number
                                    some_cost = cost(potential_board)
                                    verbose_print(
                                        f"Choosing number: {number} would result to cost: {some_cost}",
                                        verbose
                                    )
                                    if best_cost is None or some_cost < best_cost:
                                        best_cost = some_cost
                                        best_number = number
                                verbose_print(
                                    f"Number with lowest cost: {best_number} (cost = {best_cost})",
                                    verbose
                                )
                                nbrs_remaining.remove(best_number)
                                filled_board[x + i * 3][y + j * 3] = best_number
        return filled_board


def get_neighborhood(
    board: List[List[int]],
    mode: str,
    verbose: bool = False
) -> List[Tuple[List[int], List[int]]]:
    """ Finds the neighborhood (moves to look at) from the passed board.

    Future update to this function could ignore same value swaps.

    Args:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board. We assume that it has at least 1 missing value, and it has a
            cost of 0. Most likely the same board as the starting board.
        mode: Method describing how to get the neighborhood (moves). Must be "all_moves"
            or "only_in_squares". "all_moves" – all possible combinations of the empty
            spaces are considered as all possible moves. "only_in_squares" – moves are
            only all empty spaces swaps in every one of the 3x3s, this only makes sense
            if the board is guaranteed to have every number from 1 to 9 after the
            initial filling of the board.
        verbose: If true, details are printed.

    Returns:
        moves: A list of moves that are in the neighborhood. A move consists of two
            positions that can be swapped, both of the positions are described as the
            row and column number in the board.
    """

    verbose_print(
        "Finding the neighborhood for the passed board...",
        verbose
    )

    assert mode in [
        "all_moves",
        "only_in_squares"
    ], "ERROR: Unknown neighborhood finding mode specified!"

    verbose_print("Board:", verbose)
    print_board(board, verbose)

    if mode == "all_moves":
        verbose_print(
            "Looking at all the possible combinations of the empty spaces for moves...",
            verbose
        )
        # Will hold all possible positions: [0, 0], [0, 1], ..., [8, 8]
        possible_positions = []
        grid_size = 9
        for i in range(grid_size):
            for j in range(grid_size):
                # Include only empty spaces in the neighborhood
                if board[i][j] == 0:
                    possible_positions.append([i, j])
        verbose_print(f"Empty positions: {possible_positions}", verbose)

        # Will hold add possible two position combinations:
        #   ([0, 0], [0, 1])
        #   ([0, 0], [0, 2])
        #   ...
        #   ([8, 7], [8, 8])
        position_combinations = list(combinations(possible_positions, 2))
        verbose_print(f"Possible moves (combinations): {position_combinations}", verbose)

        return position_combinations
    elif mode == "only_in_squares":
        verbose_print(
            "Looking at possible empty spot combinations in each of the 3x3 squares...",
            verbose
        )
        all_moves = []
        for i in range(3):  # There are three 3x3 squares horizontally
            for j in range(3):  # There are three 3x3 squares vertically
                # Complete i-th j-th 3x3
                verbose_print(f"Look at the i={i}, j={j} 3x3:", verbose)
                possible_positions = []
                for x in range(3):  # All 3 horizontal places in the 3x3
                    for y in range(3):  # All 3 vertical places in the 3x3
                        # Include only empty spaces in the neighborhood
                        val = board[x + i * 3][y + j * 3]
                        if val == 0:
                            possible_positions.append([x + i * 3, y + j * 3])
                verbose_print(
                    f"Empty positions in specific 3x3: {possible_positions}",
                    verbose
                )
                position_combinations = list(combinations(possible_positions, 2))
                verbose_print(
                    f"Possible moves (combinations) in specific 3x3: {position_combinations}",
                    verbose
                )
                all_moves += position_combinations
        verbose_print(f"All moves (combinations): {all_moves}", verbose)
        return all_moves


def compare_neighborhood(
    board: List[List[int]],
    neighborhood: List[Tuple[List[int], List[int]]],
    verbose: bool = False
) -> List[int]:
    """ Gets a list of scores for all the moves in the neighborhood of the board.

    Args:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board.
        neighborhood: A list of moves that are in the neighborhood. A move consists of
            two positions that can be swapped, both of the positions are described as
            the row and column number in the board.
        verbose: If true, details are printed.

    Returns:
        scores: A list containing the cost if each of the neighborhood moves were
            separately done.
    """
    verbose_print(
        "Evaluating the cost for each of the moves in the neighborhood...",
        verbose
    )

    neighborhood_scores = []
    for two_positions in neighborhood:
        new_board = move_swap(board, two_positions)
        the_cost = cost(new_board)
        verbose_print(f"Move {two_positions} results in cost {the_cost}!", verbose)
        neighborhood_scores.append(the_cost)
    return neighborhood_scores


def move_swap(
    current_board: List[List[int]],
    move: Tuple[List[int], List[int]]
) -> List[List[int]]:
    """ Returns a board that would exist if the passed move is done on the passed board.

    Args:
        current_board: Two dimensional list of integers describing the state of the
            miracle sudoku board.
        move: Row and column indexes of two places that will be swapped.

    Returns:
        current_board: Two dimensional list of integers describing the state of the
            miracle sudoku board. It is the same as current_board, but with move
            swapped.
    """
    first_row, first_column = move[0]
    second_row, second_column = move[1]
    first_value = current_board[first_row][first_column]
    second_value = current_board[second_row][second_column]

    # Copy board
    new_board = copy.deepcopy(current_board)

    new_board[first_row][first_column] = second_value
    new_board[second_row][second_column] = first_value
    return new_board


def copy_board(board: List[List[int]]) -> List[List[int]]:
    """ Returns a deep copy of the miracle sudoku board.

    Args:
        board: Two dimensional list of integers describing the state of the miracle
            sudoku board. Board to copy.

    Returns:
        copied_board: Two dimensional list of integers describing the state of the
            miracle sudoku board. Copied board.
    """
    copied_board = copy.deepcopy(board)
    return copied_board
