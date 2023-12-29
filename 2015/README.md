# Advent of code 2015

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2015/advent_of_code_2015.ipynb)

Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2015/advent_of_code_2015.ipynb)
with Python solutions to the
[2015 Advent of Code puzzles](https://adventofcode.com/2015),
completed in April 2023,
by [Hugues Hoppe](http://hhoppe.com/).

The notebook presents both "compact" and "fast" code versions, along with data visualizations.

For the fast solutions, the cumulative time across all 25 puzzles is about 1 s on my PC.<br/>
(Some solutions use the `numba` package to jit-compile functions, which can take a few seconds.)<br/>
It seems difficult to further speed up the solutions because a key bottleneck is successive calls to MD5 hashing.

Here are some visualization results:

<p>
<a href="#day3">day3</a> <img src="results/day3.png" width="220"> &emsp;
<a href="#day6">day6</a> <img src="results/day6a.gif" width="220"> &nbsp;
  <img src="results/day6b.gif" width="220">
</p>

<p>
<a href="#day7">day7</a> <img src="results/day7.png" width="200"> &emsp;
<a href="#day14">day14</a> <img src="results/day14.gif" width="500">
</p>

<p>
<a href="#day18">day18</a> <img src="results/day18a.gif" width="180"> &nbsp;
  <img src="results/day18b.gif" width="180"> &emsp;
<a href="#day19">day19</a> <img src="results/day19.png" width="256">
</p>
