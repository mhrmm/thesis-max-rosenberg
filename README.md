# thesis-max-rosenberg

### February 17 2020: Added Puzzle Creation

Added functions `get_random_word`, `create_puzzle_given_root`, `create_random_puzzle`, `generate_puzzles`, and `show_puzzles`.

Notes: `create_puzzle_given_root` runs much faster than `create_random_puzzle`, as `create_random_puzzle` searches through the wordnet for a possible category then selects it and evaluates and checks that its specificity is in bound. `create_puzzle_given_root` as the name suggests is given that category/root as a parameter of the function. 

`create_random_puzzle` also generates puzzles that are sometimes too difficult even for a human player, something that is hard to correct for with only specificity as a metric. Current implementation uses `create_random_puzzle` rather than `create_puzzle_given_root`.
