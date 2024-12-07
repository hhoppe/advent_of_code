#!/usr/bin/env python3

part1 = """\
# %% [markdown]
# <a name="day{DAY}"></a>
# ## Day {DAY}: x

# %% [markdown]
# - Part 1: ?
#
# - Part 2: ?
#
# ---
#
# .

# %%
puzzle = advent.puzzle(day={DAY})

# %%
s1 = \"\"\"\\
\"\"\"


# %%
def day{DAY}(s, *, part2=False):
  pass


check_eq(day{DAY}(s1), 1)
# puzzle.verify(1, day{DAY})
"""

part2 = """\


# %%
def day{DAY}_part2(s):
  pass


check_eq(day{DAY}_part2(s1), 1)
# puzzle.verify(2, day{DAY}_part2)

# %%
day{DAY}_part2 = functools.partial(day{DAY}, part2=True)
check_eq(day{DAY}_part2(s1), 1)
# puzzle.verify(2, day{DAY}_part2)
"""

part2_day25 = """\

# %%
puzzle.verify(2, lambda s: '')  # (No "Part 2" on last day.)
"""

for day in range(1, 26):
  s = part1 + (part2_day25 if day == 25 else part2)
  s2 = s.replace('{DAY}', str(day))
  print(s2)
