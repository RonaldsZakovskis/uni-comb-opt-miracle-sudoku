# Starting version that was present in the article and the YouTube video
MINIMAL_BOARD = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# The solution for the version above
FULL_BOARD = [
    [4, 8, 3, 7, 2, 6, 1, 5, 9],
    [7, 2, 6, 1, 5, 9, 4, 8, 3],
    [1, 5, 9, 4, 8, 3, 7, 2, 6],
    [8, 3, 7, 2, 6, 1, 5, 9, 4],
    [2, 6, 1, 5, 9, 4, 8, 3, 7],
    [5, 9, 4, 8, 3, 7, 2, 6, 1],
    [3, 7, 2, 6, 1, 5, 9, 4, 8],
    [6, 1, 5, 9, 4, 8, 3, 7, 2],
    [9, 4, 8, 3, 7, 2, 6, 1, 5],
]

# But there are definitely a lot of others, for example, a few easy ones are:
#   1. Use the same board, but just reverse the values, that is, 9 -> 1, 8 -> 2, ... 1 -> 9. To get the board just go
#     through each of the values and run "value_reversed = 10 - value".
#   2. Rotate the board 90°, 180° or 270°, as each of the rules don't change their value on rotation.
#   ...
