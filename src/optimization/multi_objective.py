"""
Multi-Objective Optimization Module

Pareto-optimal optimization algorithms and utilities.

Author: Mohamed Noorul Naseem
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Solution:
    """Single solution in objective space."""
    params: Tuple[float, float, float]  # (m, p, t)
    objectives: Dict[str, float]        # {ld, cl_max, cm, mfg}
    rank: int = 0                       # Pareto rank
    crowding_distance: float = 0.0


class ParetoFront:
    """
    Pareto front management for multi-objective optimization.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.solutions: List[Solution] = []
        self.objective_names = ['ld', 'cl_max', 'stability', 'manufacturing']
    
    def add(self, params: Tuple, objectives: Dict) -> bool:
        """Add solution to front if non-dominated."""
        new_sol = Solution(params=params, objectives=objectives)
        
        # Check if dominated by any existing solution
        for sol in self.solutions:
            if self._dominates(sol.objectives, objectives):
                return False  # New solution is dominated
        
        # Remove solutions dominated by new one
        self.solutions = [
            sol for sol in self.solutions
            if not self._dominates(objectives, sol.objectives)
        ]
        
        self.solutions.append(new_sol)
        
        # Trim to max size
        if len(self.solutions) > self.max_size:
            self._update_crowding_distance()
            self.solutions.sort(key=lambda s: s.crowding_distance, reverse=True)
            self.solutions = self.solutions[:self.max_size]
        
        return True
    
    def _dominates(self, obj_a: Dict, obj_b: Dict) -> bool:
        """Check if obj_a dominates obj_b (all >= and at least one >)."""
        at_least_one_better = False
        
        for name in self.objective_names:
            val_a = self._get_objective_value(obj_a, name)
            val_b = self._get_objective_value(obj_b, name)
            
            if val_a < val_b:
                return False
            if val_a > val_b:
                at_least_one_better = True
        
        return at_least_one_better
    
    def _get_objective_value(self, obj: Dict, name: str) -> float:
        """Get normalized objective value (all maximized)."""
        if name == 'stability':
            # Minimize magnitude of cm -> maximize negative
            return -abs(obj.get('cm', 0))
        return obj.get(name, 0)
    
    def _update_crowding_distance(self):
        """Update crowding distance for diversity."""
        n = len(self.solutions)
        if n <= 2:
            for sol in self.solutions:
                sol.crowding_distance = float('inf')
            return
        
        for sol in self.solutions:
            sol.crowding_distance = 0
        
        for name in self.objective_names:
            # Sort by objective
            sorted_sols = sorted(
                self.solutions,
                key=lambda s: self._get_objective_value(s.objectives, name)
            )
            
            # Boundary solutions get infinite distance
            sorted_sols[0].crowding_distance = float('inf')
            sorted_sols[-1].crowding_distance = float('inf')
            
            # Compute distance for middle solutions
            obj_range = (
                self._get_objective_value(sorted_sols[-1].objectives, name) -
                self._get_objective_value(sorted_sols[0].objectives, name)
            )
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    sorted_sols[i].crowding_distance += (
                        self._get_objective_value(sorted_sols[i + 1].objectives, name) -
                        self._get_objective_value(sorted_sols[i - 1].objectives, name)
                    ) / obj_range
    
    def get_best(self, objective: str = 'ld') -> Optional[Solution]:
        """Get best solution for given objective."""
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda s: self._get_objective_value(s.objectives, objective))
    
    def get_compromise(self) -> Optional[Solution]:
        """Get compromise solution (best average normalized rank)."""
        if not self.solutions:
            return None
        
        # Normalize each objective and sum
        def score(sol):
            total = 0
            for name in self.objective_names:
                vals = [self._get_objective_value(s.objectives, name) for s in self.solutions]
                if max(vals) - min(vals) > 0:
                    normalized = (self._get_objective_value(sol.objectives, name) - min(vals)) / (max(vals) - min(vals))
                else:
                    normalized = 0.5
                total += normalized
            return total
        
        return max(self.solutions, key=score)
    
    def to_dict(self) -> List[Dict]:
        """Export Pareto front as list of dicts."""
        return [
            {
                'params': {'m': sol.params[0], 'p': sol.params[1], 't': sol.params[2]},
                'objectives': sol.objectives,
                'crowding_distance': sol.crowding_distance
            }
            for sol in self.solutions
        ]


class WeightedSumScalarizer:
    """Convert multi-objective to single objective using weights."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'ld': 0.40,
            'cl_max': 0.25,
            'stability': 0.20,
            'manufacturing': 0.15
        }
        
        # Normalization reference values
        self.references = {
            'ld': 50.0,
            'cl_max': 1.5,
            'stability': 0.1,  # |cm| reference
            'manufacturing': 1.0
        }
    
    def scalarize(self, objectives: Dict) -> float:
        """Convert objectives to single scalar reward."""
        score = 0.0
        
        for name, weight in self.weights.items():
            if name == 'stability':
                # Want small |cm|
                normalized = 1.0 - min(abs(objectives.get('cm', 0)) / self.references[name], 1.0)
            else:
                normalized = min(objectives.get(name, 0) / self.references[name], 1.5)
            
            score += weight * normalized
        
        return score
    
    def set_weights(self, weights: Dict[str, float]):
        """Update weights (must sum to 1)."""
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}


class EpsilonConstraint:
    """Epsilon-constraint method for multi-objective optimization."""
    
    def __init__(
        self,
        primary_objective: str = 'ld',
        constraints: Dict[str, float] = None
    ):
        self.primary = primary_objective
        self.constraints = constraints or {
            'cl_max': 1.0,
            'manufacturing': 0.7
        }
    
    def is_feasible(self, objectives: Dict) -> bool:
        """Check if solution satisfies epsilon constraints."""
        for name, min_val in self.constraints.items():
            if name == 'stability':
                if abs(objectives.get('cm', 0)) > min_val:
                    return False
            else:
                if objectives.get(name, 0) < min_val:
                    return False
        return True
    
    def evaluate(self, objectives: Dict) -> float:
        """Return primary objective if feasible, else large penalty."""
        if self.is_feasible(objectives):
            return objectives.get(self.primary, 0)
        else:
            return -1000.0  # Infeasible penalty


def compute_hypervolume(solutions: List[Solution], reference: Dict) -> float:
    """Compute hypervolume indicator for solution quality."""
    # Simplified 2D hypervolume
    if len(solutions) < 2:
        return 0.0
    
    # Project to 2D (ld, cl_max)
    points = [(s.objectives.get('ld', 0), s.objectives.get('cl_max', 0)) for s in solutions]
    points.sort(key=lambda p: p[0])
    
    ref = (reference.get('ld', 0), reference.get('cl_max', 0))
    
    hv = 0.0
    prev_y = ref[1]
    
    for x, y in points:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    
    return hv


if __name__ == "__main__":
    print("Testing Multi-Objective Module...")
    
    # Test Pareto front
    pf = ParetoFront()
    
    # Add some solutions
    pf.add((0.02, 0.4, 0.12), {'ld': 30, 'cl_max': 1.2, 'cm': -0.03, 'manufacturing': 0.9})
    pf.add((0.03, 0.4, 0.12), {'ld': 35, 'cl_max': 1.3, 'cm': -0.05, 'manufacturing': 0.8})
    pf.add((0.02, 0.4, 0.14), {'ld': 28, 'cl_max': 1.1, 'cm': -0.02, 'manufacturing': 1.0})
    
    print(f"Pareto front size: {len(pf.solutions)}")
    
    best_ld = pf.get_best('ld')
    print(f"Best L/D: {best_ld.objectives['ld']:.1f} at {best_ld.params}")
    
    compromise = pf.get_compromise()
    print(f"Compromise: {compromise.objectives}")
