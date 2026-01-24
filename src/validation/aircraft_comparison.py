"""
Compare optimized airfoil to real commercial aircraft
Calculate real-world fuel savings
"""

import json
import numpy as np
from pathlib import Path
from src.aerodynamics.xfoil_interface import XFOILRunner
from src.aerodynamics.airfoil_gen import generate_naca_4digit

class AircraftComparator:
    """
    Compare your airfoil to real aircraft and calculate savings
    """
    
    def __init__(self, database_path="data/aircraft_database.json"):
        with open(database_path, 'r') as f:
            self.aircraft_db = json.load(f)
    
    def compare_to_aircraft(self, optimized_params, aircraft_name="Boeing 737-800"):
        """
        Compare optimized airfoil to real aircraft wing section
        
        Args:
            optimized_params: [m, p, t] from RL
            aircraft_name: Aircraft to compare against
            
        Returns:
            dict with comparison results and fuel savings
        """
        
        if aircraft_name not in self.aircraft_db:
            raise ValueError(f"Aircraft {aircraft_name} not in database")
        
        aircraft = self.aircraft_db[aircraft_name]
        
        print(f"\n{'='*70}")
        print(f"COMPARISON: Your Airfoil vs {aircraft_name}")
        print(f"{'='*70}")
        
        # Get baseline airfoil from aircraft
        baseline_airfoil = aircraft['wing']['root_airfoil']
        baseline_params = [
            baseline_airfoil['m'],
            baseline_airfoil['p'],
            baseline_airfoil['t']
        ]
        
        print(f"\nBaseline ({aircraft_name} wing root):")
        print(f"  Airfoil: {aircraft['wing']['airfoil']}")
        print(f"  Parameters: m={baseline_params[0]:.3f}, p={baseline_params[1]:.3f}, t={baseline_params[2]:.3f}")
        
        print(f"\nYour RL-Optimized Airfoil:")
        print(f"  Parameters: m={optimized_params[0]:.3f}, p={optimized_params[1]:.3f}, t={optimized_params[2]:.3f}")
        
        # Run XFOIL analysis at cruise conditions
        cruise_mach = aircraft['performance']['cruise_mach']
        reynolds = self._calculate_reynolds(cruise_mach, aircraft)
        
        print(f"\nCruise Conditions:")
        print(f"  Mach: {cruise_mach}")
        print(f"  Reynolds: {reynolds:.2e}")
        print(f"  Altitude: {aircraft['performance']['cruise_altitude_ft']:,} ft")
        
        # Analyze both airfoils
        xfoil = XFOILRunner(reynolds=reynolds, mach=cruise_mach)
        
        print(f"\nRunning XFOIL CFD analysis...")
        
        # Baseline
        baseline_coords = generate_naca_4digit(*baseline_params, n_points=100)
        baseline_results = xfoil.analyze_airfoil(baseline_coords, alpha_range=[2, 3, 4, 5, 6])
        
        # Optimized
        optimized_coords = generate_naca_4digit(*optimized_params, n_points=100)
        optimized_results = xfoil.analyze_airfoil(optimized_coords, alpha_range=[2, 3, 4, 5, 6])
        
        xfoil.cleanup()
        
        if not baseline_results or not optimized_results:
            print("✗ XFOIL analysis failed")
            return None
        
        # Get cruise performance (typically α=4°)
        baseline_cruise = next((r for r in baseline_results if abs(r['alpha'] - 4.0) < 0.1), baseline_results[2])
        optimized_cruise = next((r for r in optimized_results if abs(r['alpha'] - 4.0) < 0.1), optimized_results[2])
        
        # Calculate improvements
        improvements = self._calculate_improvements(baseline_cruise, optimized_cruise)
        
        # Calculate fuel savings
        savings = self._calculate_fuel_savings(improvements, aircraft)
        
        # Display results
        self._display_results(baseline_cruise, optimized_cruise, improvements, savings, aircraft_name)
        
        return {
            'aircraft': aircraft_name,
            'baseline_params': baseline_params,
            'optimized_params': optimized_params,
            'baseline_performance': baseline_cruise,
            'optimized_performance': optimized_cruise,
            'improvements': improvements,
            'fuel_savings': savings
        }
    
    def _calculate_reynolds(self, mach, aircraft):
        """
        Calculate Reynolds number at cruise
        
        Typical values:
        - Boeing 737 at cruise: Re ≈ 15-20 million
        - Formula: Re = ρ * V * c / μ
        """
        # Simplified: use 15 million for commercial jets
        return 15e6
    
    def _calculate_improvements(self, baseline, optimized):
        """Calculate performance improvements"""
        
        baseline_ld = baseline['cl'] / baseline['cd']
        optimized_ld = optimized['cl'] / optimized['cd']
        
        improvements = {
            'cl_change_percent': ((optimized['cl'] - baseline['cl']) / baseline['cl']) * 100,
            'cd_reduction_percent': ((baseline['cd'] - optimized['cd']) / baseline['cd']) * 100,
            'ld_improvement_percent': ((optimized_ld - baseline_ld) / baseline_ld) * 100,
            'baseline_ld': baseline_ld,
            'optimized_ld': optimized_ld
        }
        
        return improvements
    
    def _calculate_fuel_savings(self, improvements, aircraft):
        """
        Calculate real-world fuel and cost savings
        
        Assumptions:
        - Drag reduction ≈ L/D improvement * factor
        - Fuel consumption proportional to drag
        - Conservative estimate (60% of theoretical improvement)
        """
        
        econ = aircraft['economics']
        perf = aircraft['performance']
        
        # Conservative drag reduction estimate
        # (Not all L/D improvement translates to fuel savings)
        effective_drag_reduction = improvements['ld_improvement_percent'] * 0.6 / 100
        
        # Per aircraft calculations
        annual_fuel_kg = perf['typical_fuel_consumption_kg_hr'] * econ['annual_utilization_hours']
        annual_fuel_savings_kg = annual_fuel_kg * effective_drag_reduction
        annual_cost_savings = annual_fuel_savings_kg * econ['fuel_price_usd_per_kg']
        
        # Lifetime savings
        lifetime_savings = annual_cost_savings * econ['aircraft_lifetime_years']
        
        # Fleet savings (if applicable)
        if 'fleet' in aircraft:
            # Use conservative number (e.g., 500 aircraft for major operator)
            fleet_size_conservative = min(500, aircraft['fleet']['in_service_worldwide'])
            fleet_savings = lifetime_savings * fleet_size_conservative
        else:
            fleet_size_conservative = 0
            fleet_savings = 0
        
        # CO2 emissions reduction
        # Jet fuel: ~3.16 kg CO2 per kg fuel
        co2_reduction_kg_per_year = annual_fuel_savings_kg * 3.16
        co2_reduction_tons_per_year = co2_reduction_kg_per_year / 1000
        
        return {
            'drag_reduction_percent': effective_drag_reduction * 100,
            'annual_fuel_savings_kg': annual_fuel_savings_kg,
            'annual_cost_savings_usd': annual_cost_savings,
            'lifetime_savings_per_aircraft_usd': lifetime_savings,
            'fleet_size_conservative': fleet_size_conservative,
            'fleet_lifetime_savings_usd': fleet_savings,
            'annual_co2_reduction_tons': co2_reduction_tons_per_year
        }
    
    def _display_results(self, baseline, optimized, improvements, savings, aircraft_name):
        """Display formatted results"""
        
        baseline_ld = baseline['cl'] / baseline['cd']
        optimized_ld = optimized['cl'] / optimized['cd']
        
        print(f"\n{'='*70}")
        print("AERODYNAMIC PERFORMANCE COMPARISON")
        print(f"{'='*70}")
        
        print(f"\nAt Cruise Condition (α={baseline['alpha']:.1f}°):")
        print(f"\n  Metric          Baseline      Optimized     Improvement")
        print(f"  {'-'*60}")
        print(f"  Cl              {baseline['cl']:8.4f}    {optimized['cl']:8.4f}    {improvements['cl_change_percent']:+7.2f}%")
        print(f"  Cd              {baseline['cd']:8.6f}  {optimized['cd']:8.6f}  {improvements['cd_reduction_percent']:+7.2f}%")
        print(f"  L/D             {baseline_ld:8.2f}    {optimized_ld:8.2f}    {improvements['ld_improvement_percent']:+7.2f}%")
        
        print(f"\n{'='*70}")
        print("FUEL SAVINGS ESTIMATE")
        print(f"{'='*70}")
        
        print(f"\nEffective Drag Reduction: {savings['drag_reduction_percent']:.2f}%")
        print(f"  (Conservative: 60% of theoretical L/D improvement)")
        
        print(f"\nPer Aircraft ({aircraft_name}):")
        print(f"  Annual fuel savings:     {savings['annual_fuel_savings_kg']:>12,.0f} kg")
        print(f"  Annual cost savings:     ${savings['annual_cost_savings_usd']:>12,.0f}")
        print(f"  25-year lifetime savings: ${savings['lifetime_savings_per_aircraft_usd']:>12,.0f}")
        
        if savings['fleet_size_conservative'] > 0:
            print(f"\nFleet Impact ({savings['fleet_size_conservative']} aircraft):")
            print(f"  Total 25-year savings:   ${savings['fleet_lifetime_savings_usd']:>12,.0f}")
            if savings['fleet_lifetime_savings_usd'] > 1e9:
                print(f"                           ${savings['fleet_lifetime_savings_usd']/1e9:>12.2f} billion")
        
        print(f"\nEnvironmental Impact:")
        print(f"  CO2 reduction per aircraft: {savings['annual_co2_reduction_tons']:>8,.0f} tons/year")
        if savings['fleet_size_conservative'] > 0:
            fleet_co2 = savings['annual_co2_reduction_tons'] * savings['fleet_size_conservative']
            print(f"  Fleet CO2 reduction:        {fleet_co2:>8,.0f} tons/year")
        
        print(f"\n{'='*70}")
    
    def compare_multiple_aircraft(self, optimized_params):
        """Compare to multiple aircraft types"""
        
        results = {}
        
        for aircraft_name in self.aircraft_db.keys():
            print(f"\n\n")
            result = self.compare_to_aircraft(optimized_params, aircraft_name)
            if result:
                results[aircraft_name] = result
        
        return results
    
    def generate_comparison_table(self, results):
        """Generate markdown table for README"""
        
        table = "| Aircraft | Baseline L/D | Optimized L/D | Improvement | Annual Savings |\n"
        table += "|----------|--------------|---------------|-------------|----------------|\n"
        
        for aircraft_name, result in results.items():
            baseline_ld = result['improvements']['baseline_ld']
            optimized_ld = result['improvements']['optimized_ld']
            improvement = result['improvements']['ld_improvement_percent']
            savings = result['fuel_savings']['annual_cost_savings_usd']
            
            table += f"| {aircraft_name} | {baseline_ld:.1f} | {optimized_ld:.1f} | "
            table += f"+{improvement:.1f}% | ${savings:,.0f} |\n"
        
        return table


def main():
    """
    Example usage
    """
    
    # Your RL-optimized parameters
    # (Replace with actual values from your trained model)
    optimized_params = [0.0385, 0.425, 0.135]  # Example: m, p, t
    
    # Create comparator
    comparator = AircraftComparator()
    
    # Compare to Boeing 737-800
    result = comparator.compare_to_aircraft(optimized_params, "Boeing 737-800")
    
    # Optionally compare to multiple aircraft
    # all_results = comparator.compare_multiple_aircraft(optimized_params)
    # table = comparator.generate_comparison_table(all_results)
    # print("\n\n" + table)
    
    return result


if __name__ == "__main__":
    result = main()
