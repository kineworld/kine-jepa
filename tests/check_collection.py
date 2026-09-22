"""Fail if the standard runner cannot see every test in ``tests/``.

This exists because of the defect in issue #3: six of the eight test modules here define
bare ``test_*`` functions, which ``unittest`` does not collect, so
``python -m unittest discover -s tests`` ran 5 of 28 tests and still reported ``OK`` --
straight past a test that fails every time.

Comparing a hand-maintained number against the runner would be the same class of mistake,
so the expected count is derived from the source with :mod:`ast` rather than written down
here:

* every top-level ``def test_*`` in a ``tests/test_*.py`` file is one test
* every ``def test_*`` inside a class in such a file is one test

Run directly, or from CI:

    python tests/check_collection.py
"""

import ast
import pathlib
import sys
import unittest

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def count_tests_in(path):
    """Return the number of test functions a file appears to define."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("test_"):
                count += 1
        elif isinstance(node, ast.ClassDef):
            for inner in node.body:
                if isinstance(inner, ast.FunctionDef) and inner.name.startswith("test_"):
                    count += 1
    return count


def main():
    files = sorted(TESTS_DIR.glob("test_*.py"))
    if not files:
        raise SystemExit("no tests/test_*.py files found")

    expected = sum(count_tests_in(f) for f in files)

    sys.path.insert(0, str(REPO_ROOT))
    suite = unittest.TestLoader().discover(str(TESTS_DIR), top_level_dir=str(REPO_ROOT))
    collected = suite.countTestCases()

    print("%d test files | %d test functions in source | %d collected by unittest"
          % (len(files), expected, collected))

    if collected != expected:
        missing = expected - collected
        raise SystemExit(
            "the runner collected %d tests but the source defines %d (%d invisible). "
            "A test that does not run cannot fail." % (collected, expected, missing)
        )

    print("ok: every test in tests/ is visible to the runner")


if __name__ == "__main__":
    main()
