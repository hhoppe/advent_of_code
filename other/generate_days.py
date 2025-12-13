#!/usr/bin/env python3

import sys

NUM_DAYS = 25
NUM_DAYS = 12

PART1 = """\
# %% [markdown]
# <a name="day{DAY}"></a>
# ## Day {DAY}: x

# %% [markdown]
# - Part 1: ?
#
# - Part 2: ?

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

PART2 = """\


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

PART2_LAST_DAY = """\

# %%
puzzle.verify(2, lambda s: '')  # (No "Part 2" on last day.)
"""


def main():
  if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(newline='\n')
  if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(newline='\n')

  for day in range(1, NUM_DAYS + 1):
    s = PART1 + (PART2_LAST_DAY if day == NUM_DAYS else PART2)
    s2 = s.replace('{DAY}', str(day))
    print(s2)


if __name__ == '__main__':
  main()
