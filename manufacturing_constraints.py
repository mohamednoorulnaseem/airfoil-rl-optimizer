"""
Manufacturing Constraints Module Wrapper

This module provides backward compatibility by wrapping the validation
manufacturing module.

Author: Mohamed Noorul Naseem
"""

from src.validation.manufacturing import (
    check_manufacturability,
    get_manufacturing_penalty,
    calculate_structural_score,
    calculate_cnc_machinability
)

__all__ = [
    'check_manufacturability',
    'get_manufacturing_penalty',
    'calculate_structural_score',
    'calculate_cnc_machinability'
]
