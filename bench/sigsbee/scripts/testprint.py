import time 

# ESC [ n A       # move cursor n lines up
# ESC [ n B       # move cursor n lines down
cursor_up = lambda lines: '\x1b[{0}A'.format(lines)
cursor_down = lambda lines: '\x1b[{0}B'.format(lines)

l = [
"""hello
world""",
"""this
is
a
multiline
string""",
"""foo
bar"""
]

max_lines = 0

for s in l:
    print(s)

    # count lines to reach the starting line
    lines_up = s.count('\n')+2

    # save maximum value
    if lines_up > max_lines:
        max_lines = lines_up

    # going up to the starting line
    time.sleep(2)
    print(cursor_up(lines_up))

# going down to ensure all output is preserved
print(cursor_down(max_lines))

