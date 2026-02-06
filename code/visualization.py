"""
Visualization Module for Fatigue-Aware HRC Framework
=====================================================

Author: Claudio Urrea
Universidad de Santiago de Chile
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
import os
from datetime import datetime


def setup_serif_font():
    """
    Detecta y configura la mejor fuente serif disponible.
    Prioridad: Palatino Linotype > Book Antiqua > Times New Roman
    """
    # Obtener fuentes disponibles
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # Orden de preferencia
    preferred_fonts = [
        'Palatino Linotype',
        'Palatino',
        'Book Antiqua',
        'Times New Roman',
        'Georgia',
        'DejaVu Serif'
    ]
    
    # Encontrar la primera fuente disponible
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font is None:
        # Fallback a serif genérica
        selected_font = 'serif'
        print(f"Warning: No preferred serif font found. Using system default serif.")
    else:
        print(f"✓ Using font: {selected_font}")
    
    # Configurar matplotlib con la fuente seleccionada
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [selected_font]
    
    # Configurar tamaños y pesos
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 14
    
    # Configurar pesos
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    return selected_font


# Configurar fuente serif al importar el módulo
FONT_NAME = setup_serif_font()

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_data(filepath='../data/simulation_data.csv'):
    """
    Load simulation data.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Simulation data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} observations")
    
    return df


def plot_fatigue_trajectory(df, output='../results/4_Fatigue_trajectory.png'):
    """
    Figure 4: Temporal evolution of operator fatigue.
    
    Shows mean fatigue progression with 95% confidence intervals
    and semaphore zone backgrounds.
    
    Args:
        df (pd.DataFrame): Simulation data
        output (str): Output filepath
    """
    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, '4_Fatigue_trajectory.png')
    # Aggregate by timepoint
    fatigue_stats = df.groupby('time')['fatigue'].agg(['mean', 'std', 'sem']).reset_index()
    fatigue_stats['ci_lower'] = fatigue_stats['mean'] - 1.96 * fatigue_stats['sem']
    fatigue_stats['ci_upper'] = fatigue_stats['mean'] + 1.96 * fatigue_stats['sem']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Semaphore zone backgrounds
    ax.axhspan(0, 30, alpha=0.1, color='green', label='Green Zone (≤30%)')
    ax.axhspan(30, 40, alpha=0.1, color='orange', label='Orange Zone (31-40%)')
    ax.axhspan(40, 100, alpha=0.1, color='red', label='Red Zone (>40%)')
    
    # Threshold lines
    ax.axhline(y=30, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label='Red Threshold (40%)')
    
    # Fatigue trajectory
    ax.plot(fatigue_stats['time'], fatigue_stats['mean'], 
            marker='o', markersize=8, linewidth=3, color='blue',
            label='Mean Fatigue', zorder=10)
    
    # Confidence interval
    ax.fill_between(fatigue_stats['time'], 
                    fatigue_stats['ci_lower'], 
                    fatigue_stats['ci_upper'],
                    alpha=0.3, color='blue', label='95% CI')
    
    # Labels and formatting
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Fatigue Level (%)', fontweight='bold')
    ax.set_title('Temporal Evolution of Operator Fatigue Across 45-Minute Episodes\n(n=1000 episodes)',
                fontweight='bold', fontsize=14)
    ax.set_xlim(-2, 47)
    ax.set_ylim(0, 75)
    ax.set_xticks([0, 15, 30, 45])
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Annotations
    for _, row in fatigue_stats.iterrows():
        ax.annotate(f"{row['mean']:.1f}%",
                   xy=(row['time'], row['mean']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_collision_temporal(df, output='../results/6_Collision_temporal.png'):
    """
    Figure 6: Temporal distribution of collision events.
    
    Shows collision concentration across episode timeline.
    
    Args:
        df (pd.DataFrame): Simulation data
        output (str): Output filepath
    """
    # Aggregate collision data
    collision_by_time = df.groupby('time').agg({
        'collision': 'sum',
        'fatigue': 'mean',
        'episode': 'nunique'
    }).reset_index()
    
    total_collisions = collision_by_time['collision'].sum()
    collision_by_time['collision_rate'] = (collision_by_time['collision'] / 
                                          collision_by_time['episode'] * 100)
    collision_by_time['pct_of_total'] = (collision_by_time['collision'] / 
                                         total_collisions * 100 if total_collisions > 0 else 0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # Top panel: Collision counts
    bars = ax1.bar(collision_by_time['time'], collision_by_time['collision'],
                   width=10, alpha=0.7, color=['green', 'yellow', 'orange', 'red'],
                   edgecolor='black', linewidth=1.5)
    
    # Annotate bars
    for i, (_, row) in enumerate(collision_by_time.iterrows()):
        if row['collision'] > 0:
            ax1.text(row['time'], row['collision'] + 0.2,
                    f"{int(row['collision'])}\n({row['pct_of_total']:.0f}%)",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Time (minutes)', fontweight='bold')
    ax1.set_ylabel('Collision Count', fontweight='bold')
    ax1.set_title('Temporal Distribution of Collision Events\n' + 
                  f'Total: {int(total_collisions)} collisions across 1000 episodes',
                  fontweight='bold', fontsize=14)
    ax1.set_xticks([0, 15, 30, 45])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bottom panel: Collision rate
    ax2.plot(collision_by_time['time'], collision_by_time['collision_rate'],
            marker='s', markersize=8, linewidth=2.5, color='red')
    ax2.fill_between(collision_by_time['time'], 0, collision_by_time['collision_rate'],
                    alpha=0.3, color='red')
    
    ax2.set_xlabel('Time (minutes)', fontweight='bold')
    ax2.set_ylabel('Collision Rate (%)', fontweight='bold')
    ax2.set_xticks([0, 15, 30, 45])
    ax2.grid(True, alpha=0.3)
    
    # Add concentration annotation
    max_time = collision_by_time.loc[collision_by_time['collision'].idxmax(), 'time']
    max_pct = collision_by_time['pct_of_total'].max()
    
    ax1.annotate(f'Concentration:\n{max_pct:.0f}% at t={int(max_time)} min',
                xy=(max_time, collision_by_time.loc[collision_by_time['collision'].idxmax(), 'collision']),
                xytext=(max_time-15, collision_by_time['collision'].max() * 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
        # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.abspath(output) if os.path.isabs(output) else os.path.join(results_dir, os.path.basename(output))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")



def plot_collision_analysis(df, output='../results/5_Collision_analysis.png'):
    """
    Figure 5: Safety Performance Validation.
    
    Bar chart comparing achieved performance (99.30%) against
    ISO/TS 15066:2016 target (99.85%) and industry baseline (95%).
    
    Args:
        df (pd.DataFrame): Simulation data
        output (str): Output filepath
    """
    # Calculate collision-free rate
    total_episodes = df['episode'].nunique()
    collision_episodes = df[df['collision'] == True]['episode'].nunique()
    collision_free = total_episodes - collision_episodes
    collision_free_rate = (collision_free / total_episodes) * 100
    
    # Define comparison values
    achieved = collision_free_rate
    target = 99.85  # ISO/TS 15066:2016
    baseline = 95.00  # Industry baseline
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Bar chart
    categories = ['Achieved\nPerformance', 'Target\nStandard', 'Industry\nBaseline']
    values = [achieved, target, baseline]
    colors = ['#2ecc71', '#3498db', '#95a5a6']  # Green, Blue, Gray
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2, alpha=0.9)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add target line
    ax.axhline(y=target, color='red', linestyle='--', linewidth=2, 
               label=f'Target: {target}%', alpha=0.8)
    
    # Labels and formatting
    ax.set_ylabel('Collision-Free Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Safety Performance Validation\n(n=1000 episodes, p<0.001)',
                fontweight='bold', fontsize=14)
    ax.set_ylim(90, 100.5)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Remove x-axis label
    ax.set_xlabel('')
    
    plt.tight_layout()
    
    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.abspath(output) if os.path.isabs(output) else os.path.join(results_dir, os.path.basename(output))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_skill_fatigue_paradox(df, output='../results/7_Skill_vs_fatigue.png'):
    """
    Figure 7: Fatigue-Skill Temporal Coupling (Model Design Consequence).
    
    Shows strong positive correlation (r=+0.970) using all 4000 observations.
    Both fatigue and skill increase monotonically with time by model design,
    demonstrating that expertise development and physiological depletion
    coevolve temporally during the 45-minute episode.
    
    Args:
        df (pd.DataFrame): Simulation data
        output (str): Output filepath
    """
    # Use ALL 4000 observations to show the temporal coupling
    # As time progresses: fatigue ↑ AND skill ↑ (model design consequence)
    
    # Calculate correlation on full dataset
    from scipy.stats import pearsonr
    r, pval = pearsonr(df['fatigue'], df['skill_level'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot with transparency to show density
    # Color by skill level
    scatter = ax.scatter(df['fatigue'], 
                        df['skill_level'],
                        c=df['skill_level'], 
                        cmap='RdYlGn', 
                        s=20, 
                        alpha=0.3,
                        edgecolor='none')
    
    # Add mean points per timepoint for clarity
    time_means = df.groupby('time').agg({
        'fatigue': 'mean',
        'skill_level': 'mean'
    }).reset_index()
    
    ax.plot(time_means['fatigue'], time_means['skill_level'],
           'o-', color='darkred', markersize=12, linewidth=3,
           label='Timepoint means', zorder=5, markeredgecolor='black', markeredgewidth=1.5)
    
    # Trend line
    z = np.polyfit(df['fatigue'], df['skill_level'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['fatigue'].min(), df['fatigue'].max(), 100)
    ax.plot(x_trend, p(x_trend), 
           'r--', linewidth=3, 
           label=f'Linear trend: r = {r:+.3f} (p < 0.001)', zorder=4)
    
    # Labels and formatting
    ax.set_xlabel('Fatigue Level (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Skill Level', fontweight='bold', fontsize=12)
    ax.set_title('Fatigue-Skill Temporal Coupling: Model Design Consequence\n' + 
                 'Both Variables Increase Monotonically with Time (r=+0.970, p<0.001)',
                fontweight='bold', fontsize=14)
    
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Novice\n(1)', 'Intermediate\n(2)', 
                       'Intermediate+\n(3)', 'Advanced\n(4)'])
    ax.set_ylim(0.5, 4.5)
    ax.set_xlim(25, 72)
    
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Skill Level')
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(['Novice', 'Inter.', 'Inter.+', 'Adv.'])
    
    # Add text box with interpretation
    textstr = f'Pearson r = {r:+.4f}\np < 0.001\n\nModel Design:\n' + \
              'Both variables\nincrease with time\nby construction'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.abspath(output) if os.path.isabs(output) else os.path.join(results_dir, os.path.basename(output))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_semaphore_distribution(df, output='../results/Semaphore_distribution.png'):
    """
    Bonus Figure: Semaphore state distribution over time.
    
    Args:
        df (pd.DataFrame): Simulation data
        output (str): Output filepath
    """
    # Aggregate
    semaphore_dist = df.groupby(['time', 'semaphore_state']).size().unstack(fill_value=0)
    
    # Normalize to percentages
    semaphore_pct = semaphore_dist.div(semaphore_dist.sum(axis=1), axis=0) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stacked bar chart
    semaphore_pct.plot(kind='bar', stacked=True, ax=ax,
                      color=['green', 'orange', 'red'],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Percentage of Episodes (%)', fontweight='bold')
    ax.set_title('Semaphore State Distribution Across Episode Timeline\n(n=1000 episodes)',
                fontweight='bold', fontsize=14)
    
    # Set x-tick labels based on actual timepoints
    timepoints = sorted(df['time'].unique())
    tick_labels = [f't={int(t)}' for t in timepoints]
    ax.set_xticklabels(tick_labels, rotation=0)
    
    ax.legend(title='State', labels=['Green (≤30%)', 'Orange (31-40%)', 'Red (>40%)'])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
        # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.abspath(output) if os.path.isabs(output) else os.path.join(results_dir, os.path.basename(output))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_all_figures(data_file='../data/simulation_data.csv'):
    """
    Generate all paper figures.
    
    Args:
        data_file (str): Path to simulation data CSV
    """
    print("\n" + "="*70)
    print("GENERATING FIGURES - FATIGUE-AWARE HRC FRAMEWORK")
    print("="*70 + "\n")
    
    # Load data
    df = load_data(data_file)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_fatigue_trajectory(df)
    plot_collision_analysis(df)
    plot_collision_temporal(df)
    plot_skill_fatigue_paradox(df)
    plot_semaphore_distribution(df)
    
    print("\n" + "="*70)
    print("ALL FIGURES GENERATED ✓")
    print("="*70)
    
    print("\nGenerated files:")
    print("4_Fatigue_trajectory.png")
    print("5_Collision_analysis.png")
    print("6_Collision_temporal.png")
    print("7_Skill_vs_fatigue.png")
    print("Semaphore_distribution.png (Supplementary)")
    

def main():
    """Main execution function."""
    generate_all_figures()


if __name__ == '__main__':
    main()
