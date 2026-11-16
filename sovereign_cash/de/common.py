"""Moved to ``sovereign_cash.core.cash_framework``.

This name is the same module object, so old imports (and monkeypatches) keep
working. New code imports ``sovereign_cash.core.cash_framework``.
"""
import sys

from sovereign_cash.core import cash_framework as _cash_framework

sys.modules[__name__] = _cash_framework
