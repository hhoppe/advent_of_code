# Advent of code 2021

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2021/advent_of_code_2021.ipynb)

IPython/Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2021/advent_of_code_2021.ipynb) by [Hugues Hoppe](http://hhoppe.com/) with solutions to the [2021 Advent of Code puzzles](https://adventofcode.com/2021).
Completed in December 2021.
See [reddit](https://www.reddit.com/r/adventofcode/comments/rtx354/advent_of_code_2021_notebook_of_compact_and_fast/?utm_source=share&utm_medium=web2x&context=3).

In this notebook, I explore both "compact" and "fast" code versions, along with data visualizations.

I was able to speed up all the solutions such that the [cumulative time](#timings) across all 25 puzzles is about 2 s.
(For some puzzles, I had to resort to the `numba` package to jit-compile Python functions.)

Here are some visualization results:

<a href="#day5">day5</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day05.png" height="200">
<a href="#day9">day9</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day09.png" height="200">
<a href="#day11">day11</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day11.gif" height="140">
<br/>
<a href="#day13">day13</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day13.gif" height="200">
<a href="#day15">day15</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day15.gif" height="200">
<br/>
<a href="#day17">day17</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day17.png" height="200">
<a href="#day20">day20</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day20.gif" height="200">
<br/>
<a href="#day21">day21</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day21.png" height="200">
<a href="#day23">day23</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day23.gif" height="100">
<a href="#day25">day25</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2021/results/day25.gif" height="200">
