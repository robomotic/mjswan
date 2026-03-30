"""Adapters for converting external framework types to mjswan internals.

Currently supports mjlab as a soft dependency.
"""

from .mjlab_adapter import adapt_actions, adapt_observations, adapt_terminations

__all__ = ["adapt_observations", "adapt_actions", "adapt_terminations"]
