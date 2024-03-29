# Advent of code 2016

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2016/advent_of_code_2016.ipynb)

Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2016/advent_of_code_2016.ipynb)
with Python solutions to the
[2016 Advent of Code puzzles](https://adventofcode.com/2016),
completed in April 2023,
by [Hugues Hoppe](http://hhoppe.com/).

The notebook presents both "compact" and "fast" code versions, along with data visualizations.

For the fast solutions, the cumulative time across all 25 puzzles is less than 4 s on my PC.<br/>
(Some solutions use the `numba` package to jit-compile functions, which can take a few seconds.)<br/>
It seems difficult to further speed up the solutions because a key bottleneck is successive calls to MD5 hashing.

Here are some visualization results:

<p>
day1 <img src="results/day1a.gif" width="256"> &emsp;
day8 <img src="results/day8a.gif" width="300">
</p>

<p>
day11 <img src="results/day11a.gif" width="140"> &emsp;
  <img src="results/day11b.gif" width="190">
</p>

<p>
day13 <img src="results/day13a.gif" width="256"> &emsp;
day18 <img src="results/day18.png" width="200">
</p>

<p>
day24 <img src="results/day24.gif" width="640">
</p>
