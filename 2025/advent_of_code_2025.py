# %% [markdown]
# # Advent of code 2025

# %% [markdown]
# [[**Open the notebook in Colab**]](https://colab.research.google.com/github/hhoppe/advent_of_code/blob/main/2025/advent_of_code_2025.ipynb)
#
# Jupyter [notebook](https://github.com/hhoppe/advent_of_code/blob/main/2025/advent_of_code_2025.ipynb)
# with Python solutions to the
# [2025 Advent of Code puzzles](https://adventofcode.com/2025),
# completed in December 2025,
# by [Hugues Hoppe](https://hhoppe.com/).
#
# The notebook presents both "compact" and "fast" code versions, along with data visualizations.
#
# For the fast solutions, the [cumulative time](#timings) across all 12 puzzles is less than 0.5 s on my PC.<br/>
# (Some solutions use the `numba` package to jit-compile functions, which can take a few seconds.)
#
# <p>
#  <a href="#day4">day4</a>
#  <img src="results/day04.gif" width="250">&emsp;
#  <a href="#day7">day7</a>
#  <img src="results/day07.gif" width="250">&emsp;
# </p>
#
# <p>
#  <a href="#day8">day8</a>
#  <img src="results/day08a.png" width="220">&nbsp;
#  <img src="results/day08b.png" width="220">&emsp;
#  <a href="#day9">day9</a>
#  <img src="results/day09.png" width="220">
# </p>
#
# <p>
#  <a href="#day11">day11</a>
#  <img src="results/day11.png" width="440">
# </p>

# %% [markdown]
# <a name="preamble"></a>
# ## Preamble

# %%
# !command -v ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq -y install ffmpeg) >/dev/null  # For mediapy.

# %%
# !dpkg -l | grep -q libgraphviz-dev || (apt-get -qq update && apt-get -qq -y install libgraphviz-dev) >/dev/null  # https://stackoverflow.com/a/66380001

# %%
# !pip install -q advent-of-code-hhoppe hhoppe-tools matplotlib mediapy \
#   networkx numba numpy pygraphviz scipy z3-solver

# %%
import bisect
import collections
from collections.abc import Callable
import functools
import itertools
import math
import pathlib
import re
from typing import Any

import advent_of_code_hhoppe  # https://github.com/hhoppe/advent-of-code-hhoppe/blob/main/advent_of_code_hhoppe/__init__.py
import hhoppe_tools as hh  # https://github.com/hhoppe/hhoppe-tools/blob/main/hhoppe_tools/__init__.py
import matplotlib.pyplot as plt
import mediapy as media  # https://github.com/google/mediapy/blob/main/mediapy/__init__.py
import networkx
import numba
import numpy as np
import scipy.optimize
import scipy.spatial.distance
import z3

# %%
if not media.video_is_available():
  media.show_videos = lambda *a, **kw: print('Creating video is unavailable.')

# %%
if pathlib.Path('results').is_dir():
  media.set_show_save_dir('results')

# %%
hh.start_timing_notebook_cells()

# %%
YEAR = 2025
PROFILE = 'google.Hugues_Hoppe.965276'
# PROFILE = 'github.hhoppe.1452460'
# # echo 53616... >~/.config/aocd/token  # session cookie from "adventofcode.com" (valid 1 month).

# %%
TAR_URL = f'https://github.com/hhoppe/advent_of_code/raw/main/{YEAR}/data/{PROFILE}.tar.gz'
# TAR_URL = ''
advent = advent_of_code_hhoppe.Advent(year=YEAR, tar_url=TAR_URL)

# %%
hh.adjust_jupyterlab_markdown_width()

# %% [markdown]
# ### Helper functions

# %%
check_eq = hh.check_eq

# %%
_ORIGINAL_GLOBALS = list(globals())

# %% [markdown]
# <a name="day1"></a>
# ## Day 1: Combination dial

# %% [markdown]
# - Part 1: The actual password is the number of times the dial is left pointing at 0 after any rotation in the sequence.  What's the actual password to open the door?
#
# - Part 2: Count the number of times any click causes the dial to point at 0, regardless of whether it happens during a rotation or at the end of one.

# %%
puzzle = advent.puzzle(day=1)

# %%
s1 = """\
L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
"""


# %%
def day1(s, *, part2=False, size=100, state=50):
  total = 0
  for line in s.splitlines():
    direction, abs_num_steps = line[0], int(line[1:])
    signed_num_steps = dict(L=-1, R=1)[direction] * abs_num_steps
    if part2:
      if signed_num_steps > 0:
        total += (state + signed_num_steps - 1) // size - state // size
      elif signed_num_steps < 0:
        total += (state - 1) // size - (state + signed_num_steps) // size
    state = (state + signed_num_steps) % size
    total += state == 0

  return total


check_eq(day1(s1), 3)
puzzle.verify(1, day1)

day1_part2 = functools.partial(day1, part2=True)
check_eq(day1_part2(s1), 6)
puzzle.verify(2, day1_part2)


# %% [markdown]
# <a name="day2"></a>
# ## Day 2: Repeated substrings

# %% [markdown]
# - Part 1: You can find the invalid IDs by looking for any ID which is made only of some sequence of digits repeated twice.  What do you get if you add up all of the invalid IDs?
#
# - Part 2: Now, an ID is invalid if it is made only of some sequence of digits repeated at least twice.

# %%
puzzle = advent.puzzle(day=2)

# %%
s1 = """\
11-22,95-115,998-1012,1188511880-1188511890,222220-222224,
1698522-1698528,446443-446449,38593856-38593862,565653-565659,
824824821-824824827,2121212118-2121212124
"""


# %%
# Slow string-based solution.
def day2a(s, *, part2=False):

  def is_matching(string):
    if not part2:
      half_len = len(string) // 2
      return string[:half_len] == string[half_len:]

    for part_len in range(1, len(string) // 2 + 1):
      div, mod = divmod(len(string), part_len)
      if mod == 0 and string[:part_len] * div == string:
        return True

    return False

  total = 0
  for range_ in s.strip().replace('\n', '').split(','):
    lo, hi = map(int, range_.split('-'))
    for value in range(lo, hi + 1):
      string = str(value)
      if is_matching(string):
        total += value

  return total


check_eq(day2a(s1), 1227775554)
# puzzle.verify(1, day2a)

day2a_part2 = functools.partial(day2a, part2=True)
check_eq(day2a_part2(s1), 4174379265)
# puzzle.verify(2, day2a_part2)


# %%
# Arithmetic solution.  E.g., a 6-digit value consists of a repeated digit sequence (of
# length 1, 2, or 3) iff it is divisible by 111111, 10101, or 1001, respectively.
# For each value in the range, we test if it is a multiple of any divisor.
@numba.njit
def day2b_total(start: int, stop: int, part2: bool) -> int:
  """Sum the values whose decimal digits form a repeated span."""
  num_digits = math.floor(math.log10(start)) + 1
  divisors = [
      (10**num_digits - 1) // (10**part_len - 1)
      for part_len in range(1, num_digits)
      if num_digits % part_len == 0 and (part_len * 2 == num_digits or part2)
  ]
  total = 0
  for value in range(start, stop):
    for divisor in divisors:
      if value % divisor == 0:
        total += value
        break

  return total


def day2b(s, *, part2=False):

  def split_by_num_digits(start, stop):
    """Partition an integer range into subranges with the same number of digits."""
    while start < stop:
      end = min(10 ** (math.floor(math.log10(start)) + 1), stop)
      yield start, end
      start = end

  total = 0
  for range_ in s.strip().replace('\n', '').split(','):
    lo, hi = map(int, range_.split('-'))
    for start, stop in split_by_num_digits(lo, hi + 1):
      total += day2b_total(start, stop, part2)

  return total


check_eq(day2b(s1), 1227775554)
puzzle.verify(1, day2b)

day2b_part2 = functools.partial(day2b, part2=True)
check_eq(day2b_part2(s1), 4174379265)
puzzle.verify(2, day2b_part2)


# %%
# Arithmetic solution.  E.g., a 6-digit value consists of a repeated digit sequence (of
# length 1, 2, or 3) iff it is divisible by 111111, 10101, or 1001, respectively.
# We gather the set of all divisor multiples inside the specified range.
def day2(s, *, part2=False):

  def split_by_num_digits(start, stop):
    """Partition an integer range into subranges with the same number of digits."""
    while start < stop:
      end = min(10 ** (math.floor(math.log10(start)) + 1), stop)
      yield start, end
      start = end

  total = 0
  for lo, hi in re.findall(r'(\d+)-(\d+)', s):
    for start, stop in split_by_num_digits(int(lo), int(hi) + 1):
      num_digits = math.floor(math.log10(start)) + 1
      divisors = [
          (10**num_digits - 1) // (10**p - 1)
          for p in range(1, num_digits)
          if num_digits % p == 0 and (p * 2 == num_digits or part2)
      ]
      values = set[int]()
      for d in divisors:
        values.update(range(-(-start // d) * d, (stop - 1) // d * d + 1, d))
      total += sum(values)

  return total


check_eq(day2(s1), 1227775554)
puzzle.verify(1, day2)

day2_part2 = functools.partial(day2, part2=True)
check_eq(day2_part2(s1), 4174379265)
puzzle.verify(2, day2_part2)

# %% [markdown]
# <a name="day3"></a>
# ## Day 3: Largest subsequence

# %% [markdown]
# - Part 1: For each line, compute the max integer formed by exactly two (ordered) digits from the line.  What is the sum?
#
# - Part 2: Same but using 12 ordered digits?

# %%
puzzle = advent.puzzle(day=3)

# %%
s1 = """\
987654321111111
811111111111119
234234234234278
818181911112111
"""


# %%
# O(N^2) solution; still fast.
def day3a(s, *, part2=False):
  # For each line, find largest k-length subsequence (as a number).
  k = (2, 12)[part2]
  total = 0

  for line in s.splitlines():
    digits = list(map(int, line))
    index = -1
    stop_index = len(digits) - k + 1
    value = 0
    for _ in range(k):
      index = max(range(index + 1, stop_index), key=lambda i: digits[i])
      value = value * 10 + digits[index]
      stop_index += 1

    total += value

  return total


check_eq(day3a(s1), 357)
puzzle.verify(1, day3a)

day3a_part2 = functools.partial(day3a, part2=True)
check_eq(day3a_part2(s1), 3121910778619)
puzzle.verify(2, day3a_part2)


# %%
# O(N + k) solution.
def day3(s, *, part2=False):
  # For each line, find largest k-length subsequence (as a number).
  k = (2, 12)[part2]
  total = 0

  for line in s.splitlines():
    n = len(line)
    result: list[str] = []
    for i, digit in enumerate(line):
      # Pop smaller digits if we can still form k-length result.
      while result and result[-1] < digit and len(result) + (n - i) > k:
        result.pop()
      if len(result) < k:
        result.append(digit)
    total += int(''.join(result))

  return total


check_eq(day3(s1), 357)
puzzle.verify(1, day3)

day3_part2 = functools.partial(day3, part2=True)
check_eq(day3_part2(s1), 3121910778619)
puzzle.verify(2, day3_part2)

# %% [markdown]
# <a name="day4"></a>
# ## Day 4: Cellular automaton

# %% [markdown]
# - Part 1: What is the number of defined cells with fewer than 4 (out of 8) defined neighbors?
#
# - Part 2: We iteratively remove all cells satisfying the condition in part1.  What is the total number of cells removed?

# %%
puzzle = advent.puzzle(day=4)

# %%
s1 = """\
..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.
"""


# %%
def day4a_part1(s):
  grid = np.array([list(line) for line in s.splitlines()]) == '@'
  grid = np.pad(grid, 1)
  neighbors = set(itertools.product((-1, 0, 1), repeat=2)) - {(0, 0)}
  num_neighbors = np.add.reduce([np.roll(grid, yx, axis=(0, 1)) for yx in neighbors])
  return (grid & (num_neighbors < 4)).sum()


check_eq(day4a_part1(s1), 13)
puzzle.verify(1, day4a_part1)


# %%
def day4(s, *, part2=False):
  grid = np.pad(np.array([list(line) for line in s.splitlines()]) == '@', 1)
  neighbors = set(itertools.product((-1, 0, 1), repeat=2)) - {(0, 0)}
  total_removed = 0

  while True:
    num_neighbors = np.add.reduce([np.roll(grid, yx, axis=(0, 1)) for yx in neighbors])
    to_remove = grid & (num_neighbors < 4)
    num_removed = to_remove.sum()
    total_removed += num_removed
    if num_removed == 0 or not part2:
      break
    grid[to_remove] = False

  return total_removed


check_eq(day4(s1), 13)
puzzle.verify(1, day4)

day4_part2 = functools.partial(day4, part2=True)
check_eq(day4_part2(s1), 43)
puzzle.verify(2, day4_part2)


# %%
def day4_visualize(s, rep=2, fps=10):
  grid = np.pad(np.array([list(line) for line in s.splitlines()]) == '@', 1)
  neighbors = set(itertools.product((-1, 0, 1), repeat=2)) - {(0, 0)}

  as_image = lambda grid: hh.to_image(grid.repeat(rep, 0).repeat(rep, 1), 250, 0)
  images = [as_image(grid)]

  while True:
    num_neighbors = np.add.reduce([np.roll(grid, yx, axis=(0, 1)) for yx in neighbors])
    to_remove = grid & (num_neighbors < 4)
    if not np.any(to_remove):
      break
    grid[to_remove] = False
    images.append(as_image(grid))

  images = [images[0]] * fps + images + [images[-1]] * fps
  media.show_video(images, codec='gif', fps=fps, title='day04')


day4_visualize(puzzle.input)

# %% [markdown]
# <a name="day5"></a>
# ## Day 5: Overlapping intervals

# %% [markdown]
# - Part 1: How many of the specified elements are contained in any interval?
#
# - Part 2: How many unique elements are in the union of all the intervals?

# %%
puzzle = advent.puzzle(day=5)

# %%
s1 = """\
3-5
10-14
16-20
12-18

1
5
8
11
17
32
"""


# %%
def day5(s, *, part2=False, optimized_part1=True):
  s_intervals, s_indices = s.split('\n\n')

  input_intervals = []
  for line in s_intervals.splitlines():
    start, end = map(int, line.split('-'))
    input_intervals.append((start, end + 1))

  indices = list(map(int, s_indices.splitlines()))

  # Sort the intervals and merge adjacent or overlapping intervals.
  intervals: list[tuple[int, int]] = []
  for start, stop in sorted(input_intervals):
    if not intervals or start > intervals[-1][1]:
      intervals.append((start, stop))
    elif stop > intervals[-1][1]:
      intervals[-1] = intervals[-1][0], stop

  if part2:
    return sum(stop - start for start, stop in intervals)

  if not optimized_part1:
    return sum(any(start <= index < stop for start, stop in intervals) for index in indices)

  count = 0
  for index in indices:
    i = bisect.bisect_left(intervals, index, key=lambda interval: interval[0])
    if i > 0:
      start, stop = intervals[i - 1]
      count += start <= index < stop
  return count


check_eq(day5(s1), 3)
puzzle.verify(1, day5)

day5_part2 = functools.partial(day5, part2=True)
check_eq(day5_part2(s1), 14)
puzzle.verify(2, day5_part2)


# %% [markdown]
# <a name="day6"></a>
# ## Day 6: Ops on tabular digits

# %% [markdown]
# - Part 1: Apply multiplication or addition on each column of numbers.  What is the total of these results?
#
# - Part 2: Each number is now represented by the digits in a column.  What is the total of the results?

# %%
puzzle = advent.puzzle(day=6)

# %%
s1 = """\
123 328  51 64 EOL
 45 64  387 23 EOL
  6 98  215 314EOL
*   +   *   +  EOL
""".replace(
    'EOL', ''
)


# %%
def day6_part1(s):
  lines = s.splitlines()
  array = np.array(list(line.split() for line in lines[:-1]), int)
  operations = lines[-1].split()
  total = 0

  ops: dict[str, Callable[..., Any]] = {'*': np.prod, '+': np.sum}
  for column, operation in zip(array.T, operations):
    total += ops[operation](column)

  return total


check_eq(day6_part1(s1), 4277556)
puzzle.verify(1, day6_part1)


# %%
def day6_part2(s):
  lines = [line + ' ' for line in s.splitlines()]
  assert all(len(line) == len(lines[0]) for line in lines)
  ops: dict[str, Callable[..., Any]] = {'*': np.prod, '+': np.sum}
  op: Callable[..., Any] | None = None
  values: list[int] = []
  total = 0

  for i, ch in enumerate(lines[-1]):
    if ch != ' ':
      op = ops[ch]
    digits = [digit for line in lines[:-1] if (digit := line[i]) != ' ']
    if digits:
      values.append(int(''.join(digits)))
    else:
      assert op
      total += op(values)
      values = []

  return total


check_eq(day6_part2(s1), 3263827)
puzzle.verify(2, day6_part2)

# %% [markdown]
# <a name="day7"></a>
# ## Day 7: Beam splitter

# %% [markdown]
# - Part 1: How many times will the beam be split?
#
# - Part 2: In total, how many different timelines would a single tachyon particle end up on?

# %%
puzzle = advent.puzzle(day=7)

# %%
s1 = """\
.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............
"""


# %%
def day7_part1(s):
  grid = np.array([list(line) for line in s.splitlines()])
  ((start_y, start_x),) = np.argwhere(grid == 'S')
  set_x = {start_x}
  num_splits = 0

  for y in range(start_y + 1, grid.shape[0]):
    new_set_x = set()

    for x in set_x:
      if grid[y, x] == '^':
        assert 0 < x < grid.shape[1] - 1
        new_set_x |= {x - 1, x + 1}
        num_splits += 1
      else:
        new_set_x |= {x}

    set_x = new_set_x

  return num_splits


check_eq(day7_part1(s1), 21)
puzzle.verify(1, day7_part1)


# %%
def day7_part2(s):
  grid = np.array([list(line) for line in s.splitlines()])
  ((start_y, start_x),) = np.argwhere(grid == 'S')
  Counter = collections.Counter[int]
  counter = Counter({start_x: 1})

  for y in range(start_y + 1, grid.shape[0]):
    new_counter = Counter()

    for x, count in counter.items():
      if grid[y, x] == '^':
        assert 0 < x < grid.shape[1] - 1
        new_counter += Counter({x - 1: count, x + 1: count})
      else:
        new_counter += Counter({x: count})

    counter = new_counter

  return counter.total()


check_eq(day7_part2(s1), 40)
puzzle.verify(2, day7_part2)


# %%
def day7_visualize(s, rep=3, fps=30):
  grid = np.array([list(line) for line in s.splitlines()])
  ((start_y, start_x),) = np.argwhere(grid == 'S')
  Counter = collections.Counter[int]
  counter = Counter({start_x: 1})
  cmap = {'.': (245, 245, 245), 'S': (255, 0, 0), '^': (30, 30, 30)}
  image = np.array([cmap[e] for e in grid.flat], np.uint8).reshape(*grid.shape, 3)
  images = [image.copy()]

  for y in range(start_y + 1, grid.shape[0]):
    new_counter = Counter()

    for x, count in counter.items():
      if grid[y, x] == '^':
        assert 0 < x < grid.shape[1] - 1
        new_counter += Counter({x - 1: count, x + 1: count})
      else:
        new_counter += Counter({x: count})

    counter = new_counter
    for x, count in counter.items():
      image[y, x] = 130, 130, 240
    images.append(image.copy())

  images = [images[0]] * fps + images + [images[-1]] * fps
  images = [image.repeat(rep, 0).repeat(rep, 1) for image in images]
  media.show_video(images, codec='gif', fps=fps, title='day07')


day7_visualize(puzzle.input)

# %% [markdown]
# <a name="day8"></a>
# ## Day 8: Components on 3D points

# %% [markdown]
# - Part 1: After inserting the 1000 shortest edges, what is the product of the sizes of the largest 3 connected components?
#
# - Part 2: Keep inserting edges by sorted length until a single component is formed.  For that final edge, what is the product of the X coordinates of the edge endpoints?

# %%
puzzle = advent.puzzle(day=8)

# %%
s1 = """\
162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689
"""


# %%
def day8(s, *, part2=False, num_edges=1000):
  points = np.array([list(map(int, line.split(','))) for line in s.splitlines()])

  # Compute all pairwise distances.
  dists = scipy.spatial.distance.pdist(points)  # Condensed edge index -> pairwise distance.
  ij_of_index = np.column_stack(np.triu_indices(len(points), k=1))  # Condensed edge index -> i, j.

  union_find = hh.UnionFind[int]()

  if not part2:
    smallest_index = np.argpartition(dists, num_edges)[:num_edges]
    smallest_index = smallest_index[np.argsort(dists[smallest_index])]

    for edge_index in smallest_index:
      i, j = ij_of_index[edge_index]
      union_find.union(i, j)

    labels = [union_find.find(i) for i in range(len(points))]
    _, counts = np.unique(labels, return_counts=True)
    return np.prod(np.sort(counts)[-3:])

  sorted_edge_indices = np.argsort(dists)
  num_components = len(points)

  for edge_index in sorted_edge_indices:
    i, j = ij_of_index[edge_index]
    label_i, label_j = union_find.find(i), union_find.find(j)
    if label_i != label_j:
      union_find.union(i, j)
      num_components -= 1
      if num_components == 1:
        return points[i][0] * points[j][0]


check_eq(day8(s1, num_edges=10), 40)
puzzle.verify(1, day8)

day8_part2 = functools.partial(day8, part2=True)
check_eq(day8_part2(s1), 25272)
puzzle.verify(2, day8_part2)


# %%
def day8_visualize(s, *, part2=False, num_edges=1000):
  points = np.array([list(map(int, line.split(','))) for line in s.splitlines()])
  dists = scipy.spatial.distance.pdist(points)
  sorted_indices = np.argsort(dists)[: None if part2 else num_edges]
  ij_of_index = np.column_stack(np.triu_indices(len(points), k=1))

  union_find = hh.UnionFind[int]()
  num_components = len(points)
  lines = []
  for edge_index in sorted_indices:
    i, j = ij_of_index[edge_index]
    lines.append([points[i], points[j]])
    label_i, label_j = union_find.find(i), union_find.find(j)
    if label_i != label_j:
      union_find.union(i, j)
      num_components -= 1
      if part2 and num_components == 1:
        break

  sorted_indices = sorted_indices[: len(lines)]
  labels = [union_find.find(i) for i in range(len(points))]
  unique_labels = list(set(labels))
  colors = hh.generate_random_colors(len(unique_labels), max_intensity=150) / 255
  colors[0] = 0.3, 0.3, 0.7
  color_of_label = {label: colors[i] for i, label in enumerate(unique_labels)}

  fig = plt.figure(figsize=(6, 6), dpi=80)
  ax: Any = fig.add_subplot(111, projection='3d')  # ax: mpl_toolkits.mplot3d.axes3d.Axes3D.
  ax.ticklabel_format(style='scientific', scilimits=(0, 0))
  ax.set(xlabel='x', ylabel='y', zlabel='z')
  ax.set_aspect('equal')
  for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_alpha(0.1)

  ax.scatter(*points.T, color='black', marker='o', s=3)

  line_colors = [color_of_label[labels[ij_of_index[index][0]]] for index in sorted_indices]
  if part2:
    line_colors[-1] = 1.0, 0.4, 0.4
  from mpl_toolkits.mplot3d.art3d import Line3DCollection

  ax.add_collection(Line3DCollection(lines, colors=line_colors, linewidths=1.0, alpha=0.8))

  image = hh.bounding_crop(hh.image_from_plt(fig), (255, 255, 255), margin=5)
  plt.close(fig)
  title = 'day08b' if part2 else 'day08a'
  media.show_image(image, title=title)


day8_visualize(puzzle.input)
day8_visualize(puzzle.input, part2=True)

# %% [markdown]
# <a name="day9"></a>
# ## Day 9: Largest rectangle

# %% [markdown]
# - Part 1: Using two red tiles as opposite corners, what is the largest area of any rectangle you can make?
#
# - Part 2: The input files form a closed path.  Using two red tiles as opposite corners, what is the largest area of any rectangle you can make that does not go outside the closed path?

# %%
puzzle = advent.puzzle(day=9)

# %%
s1 = """\
7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3
"""


# %%
def day9a_part1(s):  # Simple and fast.
  points = [tuple(map(int, line.split(','))) for line in s.splitlines()]
  return max(
      (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
      for (x1, y1), (x2, y2) in itertools.combinations(points, 2)
  )


check_eq(day9a_part1(s1), 50)
puzzle.verify(1, day9a_part1)


# %%
def _circular_pairwise(iterable):
  a, b = itertools.tee(iterable)
  first = next(b, None)
  return zip(a, itertools.chain(b, [first]))


# %%
def day9b(s, *, part2=False):
  points = [tuple(map(int, line.split(','))) for line in s.splitlines()]
  pairs = [
      (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
      for (x1, y1), (x2, y2) in itertools.combinations(points, 2)
  ]

  def area(x1, y1, x2, y2):
    return (x2 - x1 + 1) * (y2 - y1 + 1)

  if not part2:
    return max(area(*pair) for pair in pairs)

  edges = [
      (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
      for (x1, y1), (x2, y2) in itertools.pairwise(points + [points[0]])
  ]
  length = lambda p: p[2] - p[0] + p[3] - p[1]
  edges = sorted(edges, key=length, reverse=True)  # Optional, provides 4x speedup.

  return max(
      area(x1, y1, x2, y2)
      for x1, y1, x2, y2 in pairs
      if all(xx1 >= x2 or xx2 <= x1 or yy1 >= y2 or yy2 <= y1 for xx1, yy1, xx2, yy2 in edges)
  )


check_eq(day9b(s1), 50)
puzzle.verify(1, day9b)

day9b_part2 = functools.partial(day9b, part2=True)
check_eq(day9b_part2(s1), 24)
puzzle.verify(2, day9b_part2)


# %%
@numba.njit
def day9_jit(points, part2):
  n = len(points)
  pairs = np.empty((n * (n - 1) // 2, 4), dtype=points.dtype)
  index = 0
  for i in range(n):
    for j in range(i + 1, n):
      x1, y1, x2, y2 = points[i, 0], points[i, 1], points[j, 0], points[j, 1]
      pairs[index] = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
      index += 1

  if not part2:
    return np.max((pairs[:, 2] - pairs[:, 0] + 1) * (pairs[:, 3] - pairs[:, 1] + 1))

  closed = np.vstack((points, points[:1]))
  edges: Any = np.empty((n, 4), dtype=points.dtype)
  edges[:, :2] = np.minimum(closed[:-1], closed[1:])
  edges[:, 2:] = np.maximum(closed[:-1], closed[1:])

  length = lambda p: p[2] - p[0] + p[3] - p[1]
  edges = sorted(edges, key=length, reverse=True)  # 4x speedup.

  def rectangle_overlaps_any_edge(x1, y1, x2, y2):
    for xx1, yy1, xx2, yy2 in edges:
      if xx1 < x2 and xx2 > x1 and yy1 < y2 and yy2 > y1:
        return True
    return False

  max_area = 0
  for x1, y1, x2, y2 in pairs:
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if area > max_area and not rectangle_overlaps_any_edge(x1, y1, x2, y2):
      max_area = area
  return max_area


def day9(s, *, part2=False):
  points = np.array([tuple(map(int, line.split(','))) for line in s.splitlines()])
  return day9_jit(points, part2)


check_eq(day9(s1), 50)  # ~7s for jit!
puzzle.verify(1, day9)

day9_part2 = functools.partial(day9, part2=True)
check_eq(day9_part2(s1), 24)
puzzle.verify(2, day9_part2)


# %%
def day9_visualize(s):
  points = np.array([tuple(map(int, line.split(','))) for line in s.splitlines()])
  closed = np.vstack((points, points[:1]))
  pairs = [
      (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
      for (x1, y1), (x2, y2) in itertools.combinations(points, 2)
  ]
  edges = [
      (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
      for (x1, y1), (x2, y2) in itertools.pairwise(closed)
  ]
  length = lambda p: p[2] - p[0] + p[3] - p[1]
  edges = sorted(edges, key=length, reverse=True)  # 4x speedup.

  _, part1_pair = max(
      ((x2 - x1 + 1) * (y2 - y1 + 1), ((x1, y1), (x2, y2))) for x1, y1, x2, y2 in pairs
  )

  _, part2_pair = max(
      ((x2 - x1 + 1) * (y2 - y1 + 1), ((x1, y1), (x2, y2)))
      for x1, y1, x2, y2 in pairs
      if all(xx1 >= x2 or xx2 <= x1 or yy1 >= y2 or yy2 <= y1 for xx1, yy1, xx2, yy2 in edges)
  )

  fig, ax = plt.subplots(figsize=(7,) * 2, dpi=80)
  ax.set_aspect('equal')
  ax.plot(*closed.T, 'k-', linewidth=0.8)
  for ((x1, y1), (x2, y2)), color in [[part1_pair, 'blue'], [part2_pair, 'green']]:
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color=color, alpha=0.3))
  image = hh.bounding_crop(hh.image_from_plt(fig), (255, 255, 255), margin=5)
  plt.close(fig)
  media.show_image(image, title='day09')


day9_visualize(puzzle.input)

# %% [markdown]
# <a name="day10"></a>
# ## Day 10: BFS and ILP

# %% [markdown]
# - Part 1: For each machine, each button toggles a subset of lights.  What is the fewest button presses required to correctly configure the indicator lights on all of the machines?
#
# - Part 2: For each machine, each button instead increments a subset of counters.  What is the fewest button presses required to correctly configure the counters on all of the machines?

# %%
puzzle = advent.puzzle(day=10)

# %%
s1 = """\
[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
"""


# %%
def day10_part1(s):  # Using breadth-first search.
  total = 0
  for line in s.splitlines():
    s_goal, *s_changes, _ = line.split()
    goal = frozenset(i for i, ch in enumerate(s_goal[1:-1]) if ch == '#')
    changes = [set(map(int, re.findall(r'\d+', s_change))) for s_change in s_changes]

    def min_step_count_using_bfs():
      state = frozenset[int]()
      states = [state]
      visited = {state}
      for step in itertools.count(1):
        assert states
        new_states = []
        for state in states:
          for change in changes:
            new_state = frozenset(state ^ change)
            if new_state not in visited:
              if new_state == goal:
                return step
              visited.add(new_state)
              new_states.append(new_state)
        states = new_states

    total += min_step_count_using_bfs()

  return total


check_eq(day10_part1(s1), 7)
puzzle.verify(1, day10_part1)


# %%
def day10a_part2(s):  # Using the z3 solver.
  # The problem is an instance of integer linear programming (ILP), with the special case
  # of matrix M being a 0-1 matrix --- this may be a covering/packing style problem.

  def solve_min_sum_z3(M, b):
    """Solve min_x(1^T * x) s.t. x ∈ ℤ^n, x >= 0, and M * x = b."""
    solver = z3.Optimize()
    x = [z3.Int(f'x{i}') for i in range(len(M[0]))]

    # Non-negativity.
    for xi in x:
      solver.add(xi >= 0)

    # Constraints: M * x = b.
    for i, row in enumerate(M):
      # solver.add(z3.Sum([c * x[j] for j, c in enumerate(row) if c]) == b[i])
      assert all(c in {0, 1} for c in row)
      solver.add(z3.Sum([x[j] for j, c in enumerate(row) if c]) == b[i])

    # Minimize sum(x).
    solver.minimize(z3.Sum(x))

    assert solver.check() == z3.sat
    model = solver.model()
    return [model[xi].as_long() for xi in x]

  total = 0
  for line in s.splitlines():
    _, *s_changes, s_goal = line.split()
    changes = [set(map(int, re.findall(r'\d+', s_change))) for s_change in s_changes]
    b = list(map(int, re.findall(r'\d+', s_goal)))
    M = [[int(i in change) for change in changes] for i in range(len(b))]
    x = solve_min_sum_z3(M, b)
    total += sum(x)

  return total


check_eq(day10a_part2(s1), 33)
puzzle.verify(2, day10a_part2)


# %%
def day10b_part2(s):  # Explore some small speedups with the z3 solver approach.
  # - Reusing the z3 solver across several lines gives 1.1x speedup.
  # - Sorting matrix columns by contribution gives 1.05x speedup.
  # - (Multiprocessing only gave another 1.1x speedup.)
  lines = s.splitlines()
  max_cols = max(len(line.split()) - 2 for line in lines)
  total = 0

  solver = z3.Optimize()
  x = z3.IntVector('x', max_cols)

  # Permanent constraints: non-negativity.
  for xi in x:
    solver.add(xi >= 0)

  for line in lines:
    _, *s_changes, s_goal = line.split()
    changes = [set(map(int, re.findall(r'\d+', s_change))) for s_change in s_changes]
    goal = list(map(int, re.findall(r'\d+', s_goal)))
    if 0:  # No effect, but reverse=False is slower.
      coverage = [sum(i in ch for i in range(len(goal))) for ch in changes]
      order = sorted(range(len(changes)), key=lambda j: coverage[j], reverse=True)
      changes = [changes[j] for j in order]
    if 1:  # About 1.05x faster.
      contribution = [sum(goal[i] for i in range(len(goal)) if i in ch) for ch in changes]
      order = sorted(range(len(changes)), key=lambda j: contribution[j], reverse=False)
      changes = [changes[j] for j in order]

    solver.push()  # Save state.

    # Per-line constraints.
    for i, target in enumerate(goal):
      solver.add(z3.Sum([x[j] for j, ch in enumerate(changes) if i in ch]) == target)

    solver.minimize(z3.Sum(x[: len(changes)]))
    assert solver.check() == z3.sat
    total += sum(solver.model()[x[j]].as_long() for j in range(len(changes)))

    solver.pop()  # Restore state, removing per-line constraints.

  return total


check_eq(day10b_part2(s1), 33)
puzzle.verify(2, day10b_part2)


# %%
# Using bifurcate method from
# https://www.reddit.com/r/adventofcode/comments/1pk87hl/2025_day_10_part_2_bifurcate_your_way_to_victory/
def day10c_part2(s):
  total = 0

  for line in s.splitlines():
    _, *s_changes, s_goal = line.split()
    goal = tuple(map(int, re.findall(r'\d+', s_goal)))

    def compute_pattern_costs(
        num_variables: int, s_changes: list[str]
    ) -> dict[tuple[int, ...], dict[tuple[int, ...], int]]:
      changes = [[int(i) for i in s_change[1:-1].split(',')] for s_change in s_changes]
      coeffs = [tuple(int(i in change) for i in range(num_variables)) for change in changes]
      num_buttons = len(coeffs)
      all_pattern_parities = itertools.product(range(2), repeat=num_variables)
      pattern_costs: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {
          pattern_parity: {} for pattern_parity in all_pattern_parities
      }

      for num_pressed_buttons in range(num_buttons + 1):
        for buttons in itertools.combinations(range(num_buttons), num_pressed_buttons):
          pattern = tuple(map(sum, zip((0,) * num_variables, *(coeffs[i] for i in buttons))))
          pattern_parity = tuple(i % 2 for i in pattern)
          if pattern not in pattern_costs[pattern_parity]:
            pattern_costs[pattern_parity][pattern] = num_pressed_buttons

      return pattern_costs

    pattern_costs = compute_pattern_costs(len(goal), s_changes)

    @functools.cache
    def get_cost_for_goal(goal: tuple[int, ...]) -> int:
      if all(i == 0 for i in goal):
        return 0
      min_cost = 10**9
      parity = tuple(i % 2 for i in goal)

      for pattern, pattern_cost in pattern_costs[parity].items():
        if all(i <= j for i, j in zip(pattern, goal)):
          new_goal = tuple((j - i) // 2 for i, j in zip(pattern, goal))
          candidate_cost = pattern_cost + 2 * get_cost_for_goal(new_goal)
          min_cost = min(min_cost, candidate_cost)

      return min_cost

    total += get_cost_for_goal(goal)

  return total


check_eq(day10c_part2(s1), 33)
puzzle.verify(2, day10c_part2)


# %%
def day10_part2(s):  # Using scipy.optimize.milp().
  # I tried multiprocessing but it misbehaved due to the multithreading already in scipy.optimize.
  total = 0

  for line in s.splitlines():
    _, *s_changes, s_goal = line.split()
    changes = [[int(i) for i in s_change[1:-1].split(',')] for s_change in s_changes]
    goal = np.array(list(map(int, re.findall(r'\d+', s_goal))), dtype=float)

    n_rows, n_cols = len(goal), len(s_changes)
    A = np.zeros((n_rows, n_cols))
    for j, change in enumerate(changes):
      for i in change:
        A[i, j] = 1.0

    # Equality constraints A x = goal.
    lc = scipy.optimize.LinearConstraint(A, goal, goal)

    # Objective: minimize sum(x).
    c = np.ones(n_cols)

    # Bounds: x_j ≥ 0, x_j ≤ +∞.
    lb = np.zeros(n_cols)
    ub = np.full(n_cols, np.inf)
    bounds = scipy.optimize.Bounds(lb, ub)

    # All variables are integers.
    integrality = np.ones(n_cols, dtype=int)

    result = scipy.optimize.milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=[lc],
        options={'disp': False},
    )
    assert result.success
    total += int(round(result.fun))

  return total


check_eq(day10_part2(s1), 33)
puzzle.verify(2, day10_part2)

# %% [markdown]
# <a name="day11"></a>
# ## Day 11: Count graph paths

# %% [markdown]
# - Part 1: How many different paths lead from `you` to `out`?
#
# - Part 2: How many paths from `svr` to `out` visit both `dac` and `fft`?

# %%
puzzle = advent.puzzle(day=11)

# %%
s1 = """\
aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out
"""

# %%
s2 = """\
svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out
"""


# %%
def day11(s, *, part2=False):
  # The solution assumes that the graph does not contain cycles!
  edges = {lhs: set(rhs.split()) for line in s.splitlines() for lhs, rhs in [line.split(': ')]}

  @functools.cache
  def get_count(node, num_special_visited):
    if node == 'out':
      return 1 if num_special_visited >= 2 else 0
    return sum(
        get_count(node2, num_special_visited + (node in {'dac', 'fft'})) for node2 in edges[node]
    )

  return get_count('svr', 0) if part2 else get_count('you', 2)


check_eq(day11(s1), 5)
puzzle.verify(1, day11)

day11_part2 = functools.partial(day11, part2=True)
check_eq(day11_part2(s2), 2)
puzzle.verify(2, day11_part2)


# %%
def day11_visualize(s):
  edges = {lhs: set(rhs.split()) for line in s.splitlines() for lhs, rhs in [line.split(': ')]}

  graph = networkx.DiGraph()
  for node1, adjacent in edges.items():
    for node2 in adjacent:
      graph.add_edge(node1, node2)

  pos = hh.graph_layout(graph, prog='dot')  # Or prog='neato'.
  pos = hh.rotate_layout_by_angle(pos, math.tau / 4)
  fig, ax = plt.subplots(figsize=(7, 14), dpi=200)
  ax.set_aspect('equal')
  gray = (0.85,) * 3
  colors = dict(you='C1', svr='C2', out='C3', dac='C4', fft='C4')
  node_color = [colors.get(node, gray) for node in graph]
  labels = {node: node for node in graph if node in colors}
  node_size = [(120 if node in labels else 15) for node in graph]
  node_args = dict(node_size=node_size, node_color=node_color, labels=labels, font_size=6)
  networkx.draw(graph, pos, **node_args, width=0.3, arrowsize=3)
  fig.tight_layout(pad=0)
  image = hh.bounding_crop(hh.image_from_plt(fig), (255, 255, 255), margin=5)
  plt.close(fig)
  media.show_image(image, title='day11')


day11_visualize(puzzle.input)

# %% [markdown]
# <a name="day12"></a>
# ## Day 12: Tetris-like packing

# %% [markdown]
# - Part 1: How many of the regions can fit all of the presents listed?
#
# - Part 2: No part 2 for last day.

# %%
puzzle = advent.puzzle(day=12)

# %%
s1 = """\
0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
"""

# %%
if 0:
  print('\n'.join(f'    {line}' for line in puzzle.input.splitlines()[:32]))
  _ = """
    0:
    ##.
    .##
    ..#

    1:
    ###
    ##.
    #..

    2:
    #.#
    #.#
    ###

    3:
    ##.
    .##
    ###

    4:
    #.#
    ###
    #.#

    5:
    ##.
    ##.
    ###

    49x36: 50 58 46 32 38 51
    46x46: 44 43 34 34 28 41
  """


# %%
# This general packing problem is extremely difficult.
# Fortunately, it seems that all the specific problem instances (except `s1`) are easy.
# We can simply count the number of occupied grid cells in each shape.
def day12(s):
  sections = s.split('\n\n')
  shapes = []

  for s_shape in sections[:-1]:
    lines = s_shape.splitlines()
    shape = np.array([list(line) for line in lines[1:]]) == '#'
    shapes.append(shape)
  cell_count_of_shape = [np.sum(shape) for shape in shapes]

  total_feasible = 0
  for region_line in sections[-1].splitlines():
    fields = list(map(int, re.findall(r'\d+', region_line)))
    dims, counts = fields[:2], fields[2:]
    assert len(counts) == len(shapes)
    sum_cells = sum(cell_count_of_shape[i] * count for i, count in enumerate(counts))
    available_cells = dims[0] * dims[1]
    fraction = sum_cells / available_cells
    feasible = fraction <= 1.0
    # hh.show(dims, counts, sum_cells, available_cells, feasible, fraction)
    assert fraction < 0.8 or fraction > 1.0, fraction  # Easy problem instance.
    total_feasible += feasible

  return total_feasible


# check_eq(day12(s1), 2)  # The testcase itself is not an easy problem instance.
puzzle.verify(1, day12)

# %%
puzzle.verify(2, lambda s: '')  # (No "Part 2" on last day.)

# %% [markdown]
# <a name="timings"></a>
# ## Timings

# %%
advent.show_times()

# %%
if 0:  # Compute min execution times over several calls.
  advent.show_times(recompute=True, repeat=3)

# %%
if 1:  # Look for unwanted pollution of namespace.
  for _name in globals().copy():
    if not (re.match(r'^(_.*|(day|Day)\d+.*|s\d+|puzzle)$', _name) or _name in _ORIGINAL_GLOBALS):
      print(_name)

# %%
if 0:  # Lint.
  hh.run('echo pyink; pyink --diff .')
  hh.run('echo autopep8; autopep8 -j8 -d .')
  hh.run('echo mypy; mypy . || true')
  hh.run('echo pylint; pylint -j8 . || true')
  hh.run('echo pytype; pytype -j auto -k .')
  hh.run(
      'echo flake8; flake8 --indent-size=2 --exclude .ipynb_checkpoints'
      ' --extend-ignore E129,E203,E302,E305,E501,E741'
  )

# %%
hh.show_notebook_cell_top_times()


# %% [markdown]
# # End


# %% [markdown]
# <!-- For Emacs:
# Local Variables:
# fill-column: 100
# End:
# -->
