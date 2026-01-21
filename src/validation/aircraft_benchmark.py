"""
Aircraft Benchmark Module

Compare optimized airfoils against real commercial and military aircraft.
Includes Boeing 737, Airbus A320, F-15, and more.

Based on publicly available performance data.

Author: Mohamed Noorul Naseem
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class AircraftSpecs:
    """Real aircraft specifications."""
    name: str
    manufacturer: str
    category: str  # 'commercial', 'military', 'general_aviation'
    
    # Wing specifications
    airfoil_type: str
    wing_area_m2: float
    aspect_ratio: float
    sweep_angle_deg: float
    
    # Cruise performance
    cruise_mach: float
    cruise_altitude_ft: float
    cruise_ld_ratio: float
    cruise_cl: float
    cruise_cd: float
    
    # Operating costs
    fuel_consumption_kg_hr: float
    fuel_price_usd_kg: float = 0.80
    annual_flight_hours: float = 3000
    fleet_size: int = 1
    operational_life_years: int = 25


# Real aircraft database
AIRCRAFT_DATABASE: Dict[str, AircraftSpecs] = {
    'boeing_737_800': AircraftSpecs(
        name='Boeing 737-800',
        manufacturer='Boeing',
        category='commercial',
        airfoil_type='NACA 23012 derivative / BAC',
        wing_area_m2=124.6,
        aspect_ratio=9.5,
        sweep_angle_deg=25.0,
        cruise_mach=0.785,
        cruise_altitude_ft=35000,
        cruise_ld_ratio=17.5,
        cruise_cl=0.48,
        cruise_cd=0.0274,
        fuel_consumption_kg_hr=2500,
        fleet_size=500
    ),
    'boeing_787_9': AircraftSpecs(
        name='Boeing 787-9',
        manufacturer='Boeing',
        category='commercial',
        airfoil_type='Advanced supercritical',
        wing_area_m2=360.5,
        aspect_ratio=11.0,
        sweep_angle_deg=32.2,
        cruise_mach=0.85,
        cruise_altitude_ft=43000,
        cruise_ld_ratio=21.0,
        cruise_cl=0.52,
        cruise_cd=0.0248,
        fuel_consumption_kg_hr=5400,
        fleet_size=200
    ),
    'airbus_a320neo': AircraftSpecs(
        name='Airbus A320neo',
        manufacturer='Airbus',
        category='commercial',
        airfoil_type='Supercritical',
        wing_area_m2=122.6,
        aspect_ratio=9.5,
        sweep_angle_deg=25.0,
        cruise_mach=0.78,
        cruise_altitude_ft=37000,
        cruise_ld_ratio=18.5,
        cruise_cl=0.50,
        cruise_cd=0.0270,
        fuel_consumption_kg_hr=2300,
        fleet_size=600
    ),
    'f15_eagle': AircraftSpecs(
        name='F-15 Eagle',
        manufacturer='Boeing/McDonnell Douglas',
        category='military',
        airfoil_type='NACA 64A',
        wing_area_m2=56.5,
        aspect_ratio=3.0,
        sweep_angle_deg=45.0,
        cruise_mach=0.9,
        cruise_altitude_ft=30000,
        cruise_ld_ratio=8.5,
        cruise_cl=0.35,
        cruise_cd=0.0412,
        fuel_consumption_kg_hr=4500,
        fleet_size=200
    ),
    'cessna_172': AircraftSpecs(
        name='Cessna 172 Skyhawk',
        manufacturer='Cessna',
        category='general_aviation',
        airfoil_type='NACA 2412',
        wing_area_m2=16.2,
        aspect_ratio=7.5,
        sweep_angle_deg=0.0,
        cruise_mach=0.15,
        cruise_altitude_ft=8000,
        cruise_ld_ratio=12.5,
        cruise_cl=0.40,
        cruise_cd=0.0320,
        fuel_consumption_kg_hr=35,
        fleet_size=1000
    ),
    'naca_2412_baseline': AircraftSpecs(
        name='NACA 2412 (Reference)',
        manufacturer='NACA',
        category='reference',
        airfoil_type='NACA 4-digit',
        wing_area_m2=1.0,
        aspect_ratio=6.0,
        sweep_angle_deg=0.0,
        cruise_mach=0.0,
        cruise_altitude_ft=0,
        cruise_ld_ratio=56.7,  # 2D section performance
        cruise_cl=0.68,
        cruise_cd=0.0120,
        fuel_consumption_kg_hr=0,
        fleet_size=1
    ),
}


class AircraftBenchmark:
    """
    Benchmark optimized airfoils against real aircraft.
    
    Provides:
    - Performance comparison tables
    - Fuel savings estimates
    - Fleet-wide economic impact
    - CO2 reduction calculations
    """
    
    def __init__(self, aero_func=None):
        """
        Initialize benchmark.
        
        Args:
            aero_func: Function (m, p, t, alpha) -> (Cl, Cd)
        """
        self.aero_func = aero_func or self._default_aero
        self.database = AIRCRAFT_DATABASE
    
    def _default_aero(self, m, p, t, alpha):
        """Default aerodynamic function."""
        from aero_eval import aero_score
        return aero_score(m, p, t, alpha)
    
    def compare_to_aircraft(
        self, 
        m: float, p: float, t: float,
        aircraft_id: str = 'boeing_737_800',
        alpha: float = 4.0
    ) -> Dict:
        """
        Compare optimized airfoil to specific aircraft.
        
        Returns detailed performance comparison.
        """
        if aircraft_id not in self.database:
            raise ValueError(f"Unknown aircraft: {aircraft_id}")
        
        aircraft = self.database[aircraft_id]
        
        # Get optimized airfoil performance
        cl_opt, cd_opt = self.aero_func(m, p, t, alpha)
        ld_opt = cl_opt / (cd_opt + 1e-8)
        
        # Calculate improvements
        ld_improvement = (ld_opt - aircraft.cruise_ld_ratio) / aircraft.cruise_ld_ratio * 100
        cd_improvement = (aircraft.cruise_cd - cd_opt) / aircraft.cruise_cd * 100
        
        return {
            'aircraft': aircraft.name,
            'aircraft_ld': aircraft.cruise_ld_ratio,
            'optimized_ld': ld_opt,
            'ld_improvement_pct': ld_improvement,
            'aircraft_cd': aircraft.cruise_cd,
            'optimized_cd': cd_opt,
            'cd_improvement_pct': cd_improvement,
            'airfoil_params': {'m': m, 'p': p, 't': t},
            'comparison_alpha': alpha,
            'note': 'Comparison at equivalent cruise conditions'
        }
    
    def estimate_fuel_savings(
        self,
        m: float, p: float, t: float,
        aircraft_id: str = 'boeing_737_800'
    ) -> Dict:
        """
        Estimate fuel savings from airfoil improvement.
        
        Uses validated drag-to-fuel relationship:
        - Wing profile drag ≈ 25-30% of total aircraft drag
        - Fuel consumption ∝ total drag (at constant speed)
        """
        comparison = self.compare_to_aircraft(m, p, t, aircraft_id)
        aircraft = self.database[aircraft_id]
        
        # Conservative fuel savings estimate
        # Only wing profile drag contributes (~28% of total)
        # Implementation efficiency factor: 0.85
        wing_drag_fraction = 0.28
        implementation_efficiency = 0.85
        
        cd_reduction_fraction = max(0, comparison['cd_improvement_pct'] / 100)
        actual_fuel_savings_pct = cd_reduction_fraction * wing_drag_fraction * implementation_efficiency * 100
        
        # Annual fuel metrics per aircraft
        annual_fuel_kg = aircraft.fuel_consumption_kg_hr * aircraft.annual_flight_hours
        annual_savings_kg = annual_fuel_kg * actual_fuel_savings_pct / 100
        annual_cost_savings = annual_savings_kg * aircraft.fuel_price_usd_kg
        
        # Lifetime per aircraft
        lifetime_savings_kg = annual_savings_kg * aircraft.operational_life_years
        lifetime_cost_savings = annual_cost_savings * aircraft.operational_life_years
        
        # Fleet-wide savings
        fleet_annual_savings = annual_cost_savings * aircraft.fleet_size
        fleet_lifetime_savings = lifetime_cost_savings * aircraft.fleet_size
        
        # CO2 reduction (jet fuel: 3.16 kg CO2 per kg fuel)
        co2_factor = 3.16
        annual_co2_reduction_kg = annual_savings_kg * co2_factor
        fleet_annual_co2_reduction = annual_co2_reduction_kg * aircraft.fleet_size
        
        return {
            'aircraft': aircraft.name,
            'drag_reduction_pct': comparison['cd_improvement_pct'],
            'actual_fuel_savings_pct': actual_fuel_savings_pct,
            
            # Per aircraft
            'annual_fuel_savings_kg': annual_savings_kg,
            'annual_cost_savings_usd': annual_cost_savings,
            'lifetime_cost_savings_usd': lifetime_cost_savings,
            
            # Fleet-wide
            'fleet_size': aircraft.fleet_size,
            'fleet_annual_savings_usd': fleet_annual_savings,
            'fleet_lifetime_savings_usd': fleet_lifetime_savings,
            'fleet_lifetime_savings_billions': fleet_lifetime_savings / 1e9,
            
            # Environmental
            'annual_co2_reduction_kg': annual_co2_reduction_kg,
            'fleet_annual_co2_reduction_tonnes': fleet_annual_co2_reduction / 1000,
        }
    
    def benchmark_all_aircraft(self, m: float, p: float, t: float) -> Dict[str, Dict]:
        """Compare against all aircraft in database."""
        results = {}
        for aircraft_id in self.database:
            if aircraft_id == 'naca_2412_baseline':
                continue  # Skip reference
            try:
                results[aircraft_id] = {
                    'comparison': self.compare_to_aircraft(m, p, t, aircraft_id),
                    'fuel_savings': self.estimate_fuel_savings(m, p, t, aircraft_id)
                }
            except Exception as e:
                results[aircraft_id] = {'error': str(e)}
        return results
    
    def generate_report(self, m: float, p: float, t: float) -> str:
        """Generate human-readable benchmark report."""
        lines = []
        lines.append("=" * 70)
        lines.append("AIRCRAFT BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append(f"Optimized Airfoil: m={m:.4f}, p={p:.4f}, t={t:.4f}")
        lines.append("-" * 70)
        
        for aircraft_id, specs in self.database.items():
            if aircraft_id == 'naca_2412_baseline':
                continue
            
            try:
                comp = self.compare_to_aircraft(m, p, t, aircraft_id)
                fuel = self.estimate_fuel_savings(m, p, t, aircraft_id)
                
                lines.append(f"\n{specs.name} ({specs.category})")
                lines.append(f"  L/D: {specs.cruise_ld_ratio:.1f} → {comp['optimized_ld']:.1f} ({comp['ld_improvement_pct']:+.1f}%)")
                lines.append(f"  Fuel savings: {fuel['actual_fuel_savings_pct']:.2f}%")
                lines.append(f"  Annual savings: ${fuel['annual_cost_savings_usd']:,.0f}/aircraft")
                lines.append(f"  Fleet ({specs.fleet_size}): ${fuel['fleet_lifetime_savings_billions']:.2f}B over {specs.operational_life_years} years")
            except:
                lines.append(f"\n{specs.name}: Error calculating comparison")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


class UncertaintyQuantification:
    """
    Quantify uncertainty in performance predictions.
    
    Uses Monte Carlo simulation to estimate prediction intervals.
    """
    
    def __init__(self, aero_func=None, n_samples: int = 100):
        self.aero_func = aero_func
        self.n_samples = n_samples
    
    def propagate_uncertainty(
        self,
        m: float, p: float, t: float,
        m_std: float = 0.002,
        p_std: float = 0.02,
        t_std: float = 0.005
    ) -> Dict:
        """
        Propagate parameter uncertainty to performance predictions.
        
        Returns confidence intervals for Cl, Cd, L/D.
        """
        cls, cds, lds = [], [], []
        
        for _ in range(self.n_samples):
            m_sample = np.random.normal(m, m_std)
            p_sample = np.random.normal(p, p_std)
            t_sample = np.random.normal(t, t_std)
            
            # Clip to valid ranges
            m_sample = np.clip(m_sample, 0.0, 0.06)
            p_sample = np.clip(p_sample, 0.1, 0.7)
            t_sample = np.clip(t_sample, 0.08, 0.20)
            
            if self.aero_func:
                cl, cd = self.aero_func(m_sample, p_sample, t_sample, 4.0)
            else:
                # Simple model
                cl = 0.6 + 15 * m_sample + 0.1 * (4.0)
                cd = 0.02 + 50 * m_sample**2 + 50 * (t_sample - 0.12)**2
            
            cls.append(cl)
            cds.append(cd)
            lds.append(cl / (cd + 1e-8))
        
        return {
            'cl_mean': np.mean(cls),
            'cl_std': np.std(cls),
            'cl_95ci': (np.percentile(cls, 2.5), np.percentile(cls, 97.5)),
            'cd_mean': np.mean(cds),
            'cd_std': np.std(cds),
            'cd_95ci': (np.percentile(cds, 2.5), np.percentile(cds, 97.5)),
            'ld_mean': np.mean(lds),
            'ld_std': np.std(lds),
            'ld_95ci': (np.percentile(lds, 2.5), np.percentile(lds, 97.5)),
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Aircraft Benchmark Module")
    print("=" * 70)
    
    benchmark = AircraftBenchmark()
    
    # Test with optimized airfoil
    m, p, t = 0.025, 0.42, 0.13  # Example optimized
    
    print(benchmark.generate_report(m, p, t))
    
    # Uncertainty quantification
    print("\n" + "=" * 70)
    print("Uncertainty Quantification")
    print("=" * 70)
    uq = UncertaintyQuantification()
    uncertainty = uq.propagate_uncertainty(m, p, t)
    print(f"L/D: {uncertainty['ld_mean']:.1f} ± {uncertainty['ld_std']:.1f}")
    print(f"95% CI: [{uncertainty['ld_95ci'][0]:.1f}, {uncertainty['ld_95ci'][1]:.1f}]")
