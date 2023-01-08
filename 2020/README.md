<a name="top"></a>
# Advent of code 2020

[[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2020/advent_of_code_2020.ipynb)

Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2020/advent_of_code_2020.ipynb)
by [Hugues Hoppe](http://hhoppe.com/) with Python solutions to the
[2020 Advent of Code puzzles](https://adventofcode.com/2020),
completed in December 2020.

I participated in the 25-day [Advent of Code](https://adventofcode.com/) for the first time this year, thanks to encouragement from colleagues, especially [Sascha Häberling](https://github.com/shaeberling).  It was great fun and provided a nice opportunity to learn more advanced Python.

In the event, many people compete to solve the puzzles as quickly as possible --- see the impressive times on the [leaderboard](https://adventofcode.com/2020/leaderboard).
My approach was much more casual, although I did aim to finish the puzzle each evening.

Later, I went back to explore more **polished and efficient solutions**.
Can the code be expressed more succinctly?
What is the fastest algorithm given the constraints of interpreted Python?
Along the way, I discovered the [`numba`](https://numba.pydata.org/) package which can JIT-compile bottleneck functions to native code;
is it practical for these problems?  Yes, it can help greatly!

This notebook is organized such that each day is self-contained and can be run on its own after the preamble.

Some **conclusions**:

- A Jupyter/IPython notebook is a great environment for exploration.
- The notebook conveniently bundles descriptions, notes, code, small test inputs, and results.
- Initially I stored puzzle inputs within the notebook itself, but this introduces clutter and runs inefficiently.
- The cloud-based CPU kernel/runtime provided by Colab works nicely.
- With the [`numba`](https://numba.pydata.org/) library (for days [11](#day11), [15](#day15), and [23](#day23)), all of this year's puzzles can be solved in 1 second or less.
- The total execution time across all 25 puzzles is about 4 s.

Here are some visualization results:

<p>
<a href="#day11">day11</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2020/results/day11a.gif" height="150">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2020/results/day11b.gif" height="150">
<a href="#day20">day20</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2020/results/day20a.png" height="150">
<img src="https://github.com/hhoppe/advent_of_code/raw/main/2020/results/day20b.png" height="150">
</p>
<p>
<a href="#day24">day24</a><img src="https://github.com/hhoppe/advent_of_code/raw/main/2020/results/day24.gif" height="150">
</p>
