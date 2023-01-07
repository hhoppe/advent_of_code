<a name="top"></a>
# Advent of code 2018

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2018/advent_of_code_2018.ipynb)

IPython/Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2018/advent_of_code_2018.ipynb) by [Hugues Hoppe](http://hhoppe.com/) with solutions to the [2018 Advent of Code puzzles](https://adventofcode.com/2018).
Mostly completed in November 2021.

In this notebook, I explore both "compact" and "fast" code versions, along with data visualizations.

I was able to speed up all the solutions such that the [cumulative time](#timings) across all 25 puzzles is about 8 s.
(For some puzzles, I had to resort to the `numba` package to jit-compile Python functions.)

Here are some visualization results:

<a href="#day3">day3</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day03.gif" height="200">
<a href="#day6">day6</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day06.gif" height="200">
<a href="#day10">day10</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day10.gif" height="150">
<br/>
<a href="#day11">day11</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day11.gif" height="250">
<a href="#day12">day12</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day12.png" height="120">
<a href="#day13">day13</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day13.gif" height="250">
<br/>
<a href="#day15">day15</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day15a.gif" height="150">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day15b.gif" height="150">
<br/>
<a href="#day17">day17</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day17.png" height="200">
<br/>
<a href="#day18">day18</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day18a.gif" height="200">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day18b.gif" height="200">
<a href="#day20">day20</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day20.png" height="250">
<br/>
<a href="#day22">day22</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2018/results/day22.gif" height="75">
