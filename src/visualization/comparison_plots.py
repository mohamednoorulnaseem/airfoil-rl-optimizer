"""
Professional visualization for aircraft comparison
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_aircraft_comparison(results, save_path='comparison_boeing_737.png'):
    """
    Create publication-quality comparison plot
    """
    
    fig = plt.figure(figsize=(16, 10))
    
    # Extract data
    aircraft_name = results['aircraft']
    baseline_perf = results['baseline_performance']
    optimized_perf = results['optimized_performance']
    improvements = results['improvements']
    savings = results['fuel_savings']
    
    # 1. L/D Comparison Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    baseline_ld = improvements['baseline_ld']
    optimized_ld = improvements['optimized_ld']
    
    bars = ax1.bar(['Baseline\n(Boeing 737)', 'RL-Optimized'], 
                   [baseline_ld, optimized_ld],
                   color=['#1f77b4', '#2ca02c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('L/D Ratio', fontsize=12)
    ax1.set_title('Aerodynamic Efficiency Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement percentage
    improvement_pct = improvements['ld_improvement_percent']
    ax1.text(0.5, max(baseline_ld, optimized_ld) * 0.95,
            f'+{improvement_pct:.1f}%',
            ha='center', fontsize=14, color='green', fontweight='bold')
    
    # 2. Drag Reduction
    ax2 = plt.subplot(2, 3, 2)
    cd_reduction = improvements['cd_reduction_percent']
    
    ax2.barh(['Drag\nCoefficient'], [cd_reduction], 
             color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Reduction (%)', fontsize=12)
    ax2.set_title('Drag Reduction', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(cd_reduction/2, 0, f'{cd_reduction:.1f}%',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # 3. Fuel Savings (Per Aircraft)
    ax3 = plt.subplot(2, 3, 3)
    
    annual_savings = savings['annual_cost_savings_usd']
    lifetime_savings = savings['lifetime_savings_per_aircraft_usd']
    
    savings_data = [annual_savings/1000, lifetime_savings/1e6]
    bars3 = ax3.bar(['Annual\nSavings\n(x$1K)', '25-Year\nSavings\n(x$1M)'],
                    savings_data,
                    color=['#ff7f0e', '#d62728'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Savings (USD)', fontsize=12)
    ax3.set_title('Fuel Cost Savings (Per Aircraft)', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Fleet Savings
    ax4 = plt.subplot(2, 3, 4)
    
    if savings['fleet_size_conservative'] > 0:
        fleet_savings_billions = savings['fleet_lifetime_savings_usd'] / 1e9
        
        ax4.bar(['Fleet\nSavings\n(500 aircraft)'], [fleet_savings_billions],
                color='purple', alpha=0.7, edgecolor='black', width=0.5)
        ax4.set_ylabel('Savings (Billion USD)', fontsize=12)
        ax4.set_title('Total Fleet Impact (25 Years)', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        ax4.text(0, fleet_savings_billions/2, 
                f'${fleet_savings_billions:.2f}B',
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    # 5. Performance Metrics Table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    table_data = [
        ['Metric', 'Baseline', 'Optimized', 'Change'],
        ['Cl', f"{baseline_perf['cl']:.4f}", f"{optimized_perf['cl']:.4f}", 
         f"+{improvements['cl_change_percent']:.1f}%"],
        ['Cd', f"{baseline_perf['cd']:.6f}", f"{optimized_perf['cd']:.6f}", 
         f"{improvements['cd_reduction_percent']:.1f}%"],
        ['L/D', f"{baseline_ld:.2f}", f"{optimized_ld:.2f}", 
         f"+{improvement_pct:.1f}%"],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold', pad=20)
    
    # 6. CO2 Reduction
    ax6 = plt.subplot(2, 3, 6)
    
    co2_per_aircraft = savings['annual_co2_reduction_tons']
    
    if savings['fleet_size_conservative'] > 0:
        co2_fleet = co2_per_aircraft * savings['fleet_size_conservative']
        
        bars6 = ax6.bar(['Per Aircraft\n(tons/year)', 'Fleet Total\n(tons/year)'],
                       [co2_per_aircraft, co2_fleet],
                       color=['#8BC34A', '#4CAF50'], alpha=0.7, edgecolor='black')
        ax6.set_ylabel('CO₂ Reduction (tons/year)', fontsize=12)
        ax6.set_title('Environmental Impact', fontsize=13, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        for bar in bars6:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'RL-Optimized Airfoil vs {aircraft_name} Baseline Wing Section',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {save_path}")
    # plt.show() # Commented out

if __name__ == "__main__":
    # Example logic or pass
    pass
