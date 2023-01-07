<a name="top"></a>
# Advent of code 2019

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2019/advent_of_code_2019.ipynb)

IPython/Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2019/advent_of_code_2019.ipynb) by [Hugues Hoppe](http://hhoppe.com/) with solutions to the [2019 Advent of Code puzzles](https://adventofcode.com/2019).
Mostly completed in March 2021.

In this notebook, I explore both "compact" and "fast" code versions, along with data visualizations.

I was able to speed up all the solutions such that the [cumulative time](#timings) across all 25 puzzles is about 5 s.
(For some puzzles, I had to resort to the `numba` package to jit-compile Python functions.)

Here are some visualization results:

<a href="#day10">day10</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day10.gif" height="150">
<a href="#day11">day11</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day11a.gif" height="200">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day11b.gif" height="40">
<br/>
<a href="#day13">day13</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day13.gif" height="150">
<a href="#day15">day15</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day15a.gif" height="200">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day15b.gif" height="200">
<br/>
<a href="#day17">day17</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day17.png" height="160">
<a href="#day18">day18</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day18a.gif" height="220">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day18b.gif" height="220">
<a href="#day19">day19</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day19b.png" height="180">
<br/>
<a href="#day20">day20</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day20a.gif" height="250">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day20b.gif" height="250">
<a href="#day24">day24</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2019/results/day24.gif" height="250">
