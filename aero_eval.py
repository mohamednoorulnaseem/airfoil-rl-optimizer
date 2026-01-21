"""
Aerodynamic Evaluation Module Wrapper

This module provides backward compatibility by wrapping the production
aerodynamics modules.

Author: Mohamed Noorul Naseem
"""

from src.aerodynamics.legacy_eval import aero_score, aero_score_multi

__all__ = ['aero_score', 'aero_score_multi']
