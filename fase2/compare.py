from __future__ import print_function  # To make print work as in Python 3

FILE_TITLE = "cp"

FILENAMES = [
    "{}_average.txt".format(FILE_TITLE),
    "{}_output.txt".format(FILE_TITLE),
    "{}_traj.xyz".format(FILE_TITLE)
]

ORIGINAL_FILENAMES = ["output/{}".format(name) for name in FILENAMES]

MATCH = zip(FILENAMES, ORIGINAL_FILENAMES)


def is_fp(value_str):
    try:
        float(value_str)
        return True
    except ValueError:
        return False


def at_most_equal(a, b, level=10):
    at = 0

    if len(a) != len(b):
        return False, 0

    for char_idx in range(len(a)):
        if a[char_idx].isdigit() and b[char_idx].isdigit():
            if a[char_idx] != b[char_idx] and at <= level:
                return False, at + 1
            at += 1

    return True, 0


def pprint(at, string, color):
    COLOR_CODE = "\033[93m" if color == "red" else "\033[92m"

    for id, char in enumerate(string):
        if id >= at:
            print("{}{}\033[00m".format(COLOR_CODE, char), end="")
        else:
            print(char, end="")
    print("\n")


def print_diff(old_val, new_val, linenum, col, at, filename):
    print("{}: Values do \033[91m not match \033[00m at line {}, column {}:".format(filename, linenum, col),
          end="")  # Add end="" to the print statement to avoid the SyntaxError
    pprint(at, new_val, "red")
    print((" " * (at + 10)) + ("^" * (len(new_val) - at)))
    print("Original: ", end="")
    pprint(at, old_val, "green")


def compare():
    for (new_file, og_file) in MATCH:
        diff_count = 0

        with open(new_file, "r") as new:
            new_lines = new.readlines()[2:]

        with open(og_file, "r") as og:
            og_lines = og.readlines()[2:]

        for i, (new, old) in enumerate(zip(new_lines, og_lines)):
            new_values = filter(lambda v: is_fp(v), new.split(" "))
            old_values = filter(lambda v: is_fp(v), old.split(" "))

            for j, (new_val, old_val) in enumerate(zip(new_values, old_values)):
                cmp, at = at_most_equal(new_val, old_val)
                if not cmp:
                    print_diff(old_val, new_val, i, j, at, new_file)
                    diff_count += 1

        print("Detected \033[91m {}\033[00m diffs on file {}".format(diff_count, new_file))
        raw_input("...Continue [ENTER]\n")


if __name__ == "__main__":
    try:
        SystemExit(compare())
    except KeyboardInterrupt:
        print("")
