# Solving "Miracle Sudoku" using Simulated Annealing

This repository contains a "Miracle Sudoku" solver that uses Simulated Annealing
optimization algorithm. The implementation is quite basic and a lot of time hasn't been
spent into making the algorithms more efficient and making the code more beautiful, for
example, splitting stuff into classes.

Backtracking seems to be faster than Simulated Annealing when solving regular Sudoku,
and as Miracle Sudoku has more rules, I assume that backtracking is even faster for this
problem, as well as Simulated Annealing is slower. So, if one is planning to create a
fast Miracle Sudoku solver, maybe they should look into backtracking and investigate
some of the conclusions reached in the article, for example, that the complete grid has
repeating symmetry, but it might be necessary to prove first that this symmetry is
always the case, though.

At the moment, code doesn't use any external libraries, so a plain installation of
Python 3.8 should suffice.

Installation instructions:
```
conda create -n uni-comb-opt-miracle-sudoku python=3.8
conda activate uni-comb-opt-miracle-sudoku
```

Running instructions:
```
python main.py
```

Delete the environment with:
```
conda deactivate
conda remove -n uni-comb-opt-miracle-sudoku --all
```
