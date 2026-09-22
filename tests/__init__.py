"""Make every ``test_*`` function in this package visible to the standard runner.

Six of the eight test modules here were written to be run directly
(``python tests/test_rollout.py``) rather than as ``unittest.TestCase`` subclasses.
``unittest discover`` collects only the latter, so it ran 5 of the 28 tests and still
reported ``OK`` -- including past a test that fails every single time.

Rather than rewrite 23 functions across six files, this module implements the
documented ``load_tests`` protocol and wraps the bare functions. The direct-run entry
point in each file keeps working unchanged.

Use either of:

    python -m unittest discover -s tests -t .
    python -m unittest tests

Note that ``-t .`` matters: without it ``discover`` treats ``tests/`` as a directory of
top-level modules and never imports this package, so ``load_tests`` would not run.

Careful: a ``load_tests`` function *replaces* the default collection for this package
rather than adding to it, so this must collect the ``TestCase`` classes itself as well
as wrapping the bare functions. Collecting only the latter would silently drop the five
``TestCase`` methods in ``test_counterfactual.py`` and ``test_posttrain.py`` -- the very
bug this module exists to fix, pointed the other way.
"""

import importlib
import inspect
import pkgutil
import unittest


def _wrap(module_name, function_name, function):
    """Return a TestCase that runs one bare ``test_*`` function."""

    class _BareFunctionTest(unittest.TestCase):
        def runTest(self):  # noqa: N802 - unittest's own protocol name
            function()

    _BareFunctionTest.__name__ = "Test_%s_%s" % (module_name, function_name)
    _BareFunctionTest.__qualname__ = _BareFunctionTest.__name__
    return _BareFunctionTest()


def _modules():
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("test_"):
            yield info.name, importlib.import_module("%s.%s" % (__name__, info.name))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for module_name, module in _modules():
        # 1. TestCase subclasses declared in the module, via the standard loader.
        suite.addTests(loader.loadTestsFromModule(module))
        # 2. Bare test_* functions, which that loader ignores.
        for name, obj in sorted(vars(module).items()):
            if name.startswith("test_") and inspect.isfunction(obj):
                suite.addTest(_wrap(module_name, name, obj))
    return suite
