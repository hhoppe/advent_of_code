# Advent of code 2023

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2023/advent_of_code_2023.ipynb)

Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2023/advent_of_code_2023.ipynb)
with Python solutions to the
[2023 Advent of Code puzzles](https://adventofcode.com/2023),
completed in December 2023,
by [Hugues Hoppe](http://hhoppe.com/).

The notebook presents both "compact" and "fast" code versions, along with data visualizations.

For the fast solutions, the cumulative time across all 25 puzzles is about 6 s on my PC.<br/>
(Some solutions use the `numba` package to jit-compile functions, which can take a few seconds.)

Here are some visualization results (obtained by setting `SHOW_BIG_MEDIA = True`):

<p>
<a href="#day3">day3</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day03a.png" width="120">&nbsp;
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day03b.png" width="120">&emsp;
<a href="#day6">day6</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day06.gif" width="220">&emsp;
<a href="#day7">day7</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day07.png" width="200">
</p>

<p>
<a href="#day10">day10</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day10a.png" width="100">&nbsp;
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day10b.png" width="100">&emsp;
<a href="#day13">day13</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day13.png" width="280">&emsp;
<a href="#day14">day14</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day14.gif" width="140">
</p>

<p>
<a href="#day16">day16</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day16a.gif" width="200">&emsp;
<a href="#day17">day17</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day17a.gif" width="110">&nbsp;
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day17b.gif" width="110">&emsp;
<a href="#day18">day18</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day18b.png" width="220">
</p>

<p>
<a href="#day20">day20</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day20.png" width="240">&emsp;
<a href="#day21">day21</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day21a.gif" width="240">&emsp;
<a href="#day22">day22</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day22a.png" width="60">&nbsp;
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day22b.gif" width="72">
</p>

<p>
<a href="#day23">day23</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day23d.png" width="180">&nbsp;
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day23b.png" width="180">&emsp;
<a href="#day25">day25</a> <img src="https://github.com/hhoppe/advent_of_code/raw/main/2023/results/day25.png" width="320">
</p>
