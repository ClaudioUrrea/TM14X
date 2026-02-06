"""
Fatigue-Aware Task Reallocation Framework for HRC
==================================================

Simulation framework testing adaptive task allocation algorithms 
in human-robot collaboration where operator fatigue accumulates 
during extended work shifts.

Paper: "A Simulation-Based Framework for Fatigue-Aware Task 
Reallocation in Human-Robot Collaboration: Software Verification 
with 1000-Episode Digital Twin"

Author: Claudio Urrea
Universidad de Santiago de Chile
claudio.urrea@usach.cl
ORCID: 0000-0001-7197-8928

Usage:
    python Dynamic_Threshold_Algorithm.py --episodes 1000 --seed 42

Dependencies: See requirements.txt

Citation:
    Urrea, C. (2025). A Simulation-Based Framework for 
    Fatigue-Aware Task Reallocation in HRC. IEEE Access.
"""

import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import os


class OperatorState:
    """
    Maintains operator physiological and skill state.
    
    Attributes:
        fatigue (float): Current fatigue level (0-100%)
        skill_level (int): Competence (1=Novice, 2=Intermediate, 3=Int+, 4=Advanced)
        current_task (str): Active task name
    """
    
    def __init__(self, initial_fatigue=28.0, initial_skill=2):
        self.fatigue = initial_fatigue
        self.skill_level = initial_skill
        self.current_task = 'rest'
        
    def update_fatigue(self, task, duration_minutes):
        """
        Update fatigue based on task and duration.
        
        Args:
            task (str): Task name
            duration_minutes (float): Time elapsed
        """
        # Task-specific fatigue rates from literature (Ma et al., Diego-Mas et al.)
        fatigue_rates = {
            'rest': -0.50,          # Recovery
            'hmi_supervision': 0.67,  # Cognitive load
            'material_delivery': 0.93,  # Physical exertion
            'packaging': 1.07       # Awkward postures
        }
        
        rate = fatigue_rates.get(task, 0.0)
        self.fatigue += rate * duration_minutes
        
        # Bound fatigue [0, 100]
        self.fatigue = max(0.0, min(100.0, self.fatigue))
        
    def progress_skill(self, timepoint):
        """
        Skill progression at checkpoints.
        
        Args:
            timepoint (int): Current time in minutes
        """
        # Deterministic skill progression
        # t=0: 2 (Intermediate baseline)
        # t=15: 2 (Intermediate)
        # t=30: 3 (Intermediate+)
        # t=45: 4 (Advanced)
        if timepoint >= 45:
            self.skill_level = 4  # Advanced
        elif timepoint >= 30:
            self.skill_level = 3  # Intermediate+
        else:
            self.skill_level = 2  # Intermediate (t=0 and t=15)


class RobotController:
    """
    Implements robotic intervention logic and collision detection.
    """
    
    def __init__(self):
        self.is_active = False
        self.intervention_count = 0
        self.active_duration = 0.0
        
    def should_intervene(self, fatigue, semaphore_state):
        """
        Decision logic for robotic takeover.
        
        Args:
            fatigue (float): Operator fatigue level
            semaphore_state (str): Current zone (green/orange/red)
            
        Returns:
            bool: True if intervention required
        """
        return semaphore_state == 'red' and fatigue > 40.0
        
    def check_collision(self, fatigue, distance, velocity):
        """
        Simplified collision detection model.
        
        Args:
            fatigue (float): Operator fatigue (%)
            distance (float): Human-robot separation (m)
            velocity (float): Robot velocity (m/s)
            
        Returns:
            bool: True if collision detected
        """
        # ISO/TS 15066 thresholds
        min_separation = 0.05  # 5 cm
        safe_velocity = 0.25   # 0.25 m/s
        
        if distance < min_separation and velocity > safe_velocity and fatigue > 40:
            return True
        return False


class FatigueMonitoringSystem:
    """
    Tricolor semaphore with adaptive sampling.
    """
    
    def __init__(self, green_threshold=30.0, red_threshold=40.0):
        self.green_threshold = green_threshold
        self.red_threshold = red_threshold
        
    def classify_zone(self, fatigue):
        """
        Determine semaphore state.
        
        Args:
            fatigue (float): Current fatigue level
            
        Returns:
            str: 'green', 'orange', or 'red'
        """
        if fatigue <= self.green_threshold:
            return 'green'
        elif fatigue <= self.red_threshold:
            return 'orange'
        else:
            return 'red'
            
    def get_sampling_rate(self, zone):
        """
        Adaptive monitoring frequency.
        
        Args:
            zone (str): Semaphore state
            
        Returns:
            float: Sampling period in minutes
        """
        rates = {
            'green': 5.0,
            'orange': 2.0,
            'red': 0.1
        }
        return rates[zone]


class SimulationManager:
    """
    Orchestrates episode execution and data logging.
    """
    
    def __init__(self, seed=42, green_threshold=30.0, red_threshold=40.0):
        np.random.seed(seed)
        self.green_threshold = green_threshold
        self.red_threshold = red_threshold
        self.episode_data = []
        
    def run_episode(self, episode_id):
        """
        Execute single 45-minute episode.
        
        Args:
            episode_id (int): Episode identifier
            
        Returns:
            list: Observation records at checkpoints
        """
        # Initialize components
        initial_fatigue = np.random.normal(28.0, 0.75)
        operator = OperatorState(initial_fatigue=initial_fatigue, initial_skill=2)
        robot = RobotController()
        monitor = FatigueMonitoringSystem(self.green_threshold, self.red_threshold)
        
        # Define checkpoints and corresponding tasks
        # Each segment has a task and duration
        segments = [
            (0, 'rest', 0),                      # t=0: initial state
            (15, 'hmi_supervision', 15),         # t=15: after 15 min of supervision
            (30, 'material_delivery', 15),       # t=30: after 15 min of delivery
            (45, 'packaging', 15)                # t=45: after 15 min of packaging
        ]
        
        observations = []
        collision_occurred = False
        
        for checkpoint_time, task_name, duration in segments:
            # Execute task for this segment (skip for t=0)
            if duration > 0:
                operator.current_task = task_name
                operator.update_fatigue(task_name, duration)
            
            # Progress skill at appropriate times
            operator.progress_skill(checkpoint_time)
            
            # Get monitoring state
            zone = monitor.classify_zone(operator.fatigue)
            sampling_rate = monitor.get_sampling_rate(zone)
            
            # Check for intervention
            if robot.should_intervene(operator.fatigue, zone):
                robot.is_active = True
                robot.intervention_count += 1
                if duration > 0:
                    robot.active_duration += duration * 0.5
            
            # Collision check (only at high fatigue)
            if operator.fatigue > 65 and np.random.random() < 0.01:
                collision_occurred = True
            
            # Record observation
            obs = {
                'episode': episode_id,
                'time': checkpoint_time,
                'task': task_name,
                'fatigue': operator.fatigue,
                'skill_level': operator.skill_level,
                'semaphore_state': zone,
                'sampling_rate': sampling_rate,
                'robot_active': robot.is_active,
                'collision': collision_occurred
            }
            observations.append(obs)
        
        return observations
        
    def run_simulation(self, num_episodes):
        """
        Execute full simulation campaign.
        
        Args:
            num_episodes (int): Number of episodes to simulate
            
        Returns:
            pd.DataFrame: Complete dataset
        """
        print(f"\n{'='*60}")
        print(f"SIMULATION CAMPAIGN: {num_episodes} Episodes")
        print(f"{'='*60}\n")
        
        all_observations = []
        
        for ep in tqdm(range(num_episodes), desc="Simulating"):
            episode_obs = self.run_episode(ep)
            all_observations.extend(episode_obs)
            
        df = pd.DataFrame(all_observations)
        
        print(f"\n✓ Simulation complete")
        print(f"  Total observations: {len(df)}")
        print(f"  Episodes: {df['episode'].nunique()}")
        print(f"  Checkpoints per episode: {len(df) // num_episodes}")
        
        return df
        
    def validate_data(self, df):
        """
        Verify simulation data integrity.
        
        Args:
            df (pd.DataFrame): Simulation data
            
        Returns:
            bool: True if validation passes
        """
        print(f"\n{'='*60}")
        print("DATA VALIDATION")
        print(f"{'='*60}\n")
        
        # Check fatigue bounds
        assert df['fatigue'].between(0, 100).all(), \
            "Fatigue out of [0,100] range"
        print("✓ Fatigue bounds [0,100]: OK")
        
        # Check skill monotonicity
        for ep in df['episode'].unique():
            skills = df[df['episode']==ep]['skill_level'].values
            assert all(skills[i] <= skills[i+1] 
                      for i in range(len(skills)-1)), \
                f"Non-monotonic skill in episode {ep}"
        print("✓ Skill monotonicity: OK")
        
        # Check semaphore consistency
        green = df[df['semaphore_state']=='green']
        if len(green) > 0:
            assert (green['fatigue'] <= self.green_threshold).all(), \
                "Green state with high fatigue"
        print(f"✓ Green zone (≤{self.green_threshold}%): OK")
        
        red = df[df['semaphore_state']=='red']
        if len(red) > 0:
            assert (red['fatigue'] > self.red_threshold).all(), \
                "Red state with low fatigue"
        print(f"✓ Red zone (>{self.red_threshold}%): OK")
        
        # Check collision count
        collision_count = df['collision'].sum()
        collision_free_rate = (df['episode'].nunique() - collision_count) / df['episode'].nunique()
        print(f"✓ Collision-free rate: {collision_free_rate*100:.2f}%")
        
        print(f"\n✓ All validation checks passed\n")
        return True
        
    def compute_summary(self, df):
        """
        Calculate aggregate statistics.
        
        Args:
            df (pd.DataFrame): Simulation data
            
        Returns:
            dict: Summary statistics
        """
        collision_count = df.groupby('episode')['collision'].max().sum()
        total_episodes = df['episode'].nunique()
        collision_free_rate = (total_episodes - collision_count) / total_episodes
        
        # Fatigue statistics by timepoint
        fatigue_stats = df.groupby('time')['fatigue'].agg(['mean', 'std', 'min', 'max']).to_dict('index')
        
        # Semaphore distribution
        semaphore_dist = df.groupby(['time', 'semaphore_state']).size().unstack(fill_value=0).to_dict('index')
        
        # Robot activity
        robot_active_pct = df['robot_active'].mean() * 100
        
        # Skill progression
        skill_progression = df.groupby('time')['skill_level'].mean().to_dict()
        
        summary = {
            'metadata': {
                'total_episodes': total_episodes,
                'observations': len(df),
                'simulation_date': datetime.now().isoformat(),
                'green_threshold': self.green_threshold,
                'red_threshold': self.red_threshold
            },
            'safety': {
                'collision_count': int(collision_count),
                'collision_free_rate': float(collision_free_rate),
                'collision_free_percentage': float(collision_free_rate * 100),
                'iso_target': 99.85,
                'gap_to_target': float(99.85 - collision_free_rate * 100)
            },
            'fatigue': {
                'by_timepoint': fatigue_stats
            },
            'semaphore': {
                'distribution': semaphore_dist
            },
            'robot': {
                'active_percentage': float(robot_active_pct)
            },
            'skill': {
                'progression': skill_progression
            }
        }
        
        return summary
        
    def save_results(self, df, summary, output_dir='../data'):
        """
        Save simulation data and summary.
        
        Args:
            df (pd.DataFrame): Simulation data
            summary (dict): Summary statistics
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'simulation_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
        
        # Save JSON summary
        json_path = os.path.join(output_dir, 'simulation_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {json_path}")
        
        print(f"\n{'='*60}")
        print("RESULTS SAVED")
        print(f"{'='*60}\n")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Fatigue-aware HRC simulation framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 1000 episodes
  python Dynamic_Threshold_Algorithm.py
  
  # Custom episodes
  python Dynamic_Threshold_Algorithm.py --episodes 100
  
  # Custom thresholds
  python Dynamic_Threshold_Algorithm.py --green-threshold 35 --red-threshold 45
        """
    )
    
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to simulate (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='../data',
                       help='Output directory for results (default: ../data)')
    parser.add_argument('--green-threshold', type=float, default=30.0,
                       help='Green zone threshold percent (default: 30.0)')
    parser.add_argument('--red-threshold', type=float, default=40.0,
                       help='Red zone threshold percent (default: 40.0)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip data validation (default: False)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Display configuration
    print(f"\n{'='*60}")
    print("FATIGUE-AWARE HRC SIMULATION FRAMEWORK")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Random seed: {args.seed}")
    print(f"  Green threshold: {args.green_threshold}%")
    print(f"  Red threshold: {args.red_threshold}%")
    print(f"  Output directory: {args.output}")
    print(f"{'='*60}\n")
    
    # Initialize simulation
    sim = SimulationManager(
        seed=args.seed,
        green_threshold=args.green_threshold,
        red_threshold=args.red_threshold
    )
    
    # Run simulation
    df = sim.run_simulation(args.episodes)
    
    # Validate
    if not args.no_validation:
        sim.validate_data(df)
    
    # Compute summary
    summary = sim.compute_summary(df)
    
    # Display key results
    print(f"{'='*60}")
    print("KEY RESULTS")
    print(f"{'='*60}")
    print(f"Safety Performance:")
    print(f"  Collision-free rate: {summary['safety']['collision_free_percentage']:.2f}%")
    print(f"  ISO/TS 15066 target: {summary['safety']['iso_target']:.2f}%")
    print(f"  Gap to target: {summary['safety']['gap_to_target']:.2f} pp")
    print(f"\nFatigue Progression:")
    for time, stats in summary['fatigue']['by_timepoint'].items():
        print(f"  t={time:2d} min: {stats['mean']:5.2f}% ± {stats['std']:4.2f}%")
    print(f"\nRobot Activity: {summary['robot']['active_percentage']:.2f}%")
    print(f"{'='*60}\n")
    
    # Save results
    sim.save_results(df, summary, args.output)
    
    print("Simulation complete! ✓")
    print(f"\nNext steps:")
    print(f"  1. Run statistical analysis: python code/statistical_analysis.py")
    print(f"  2. Generate figures: python code/visualization.py")


if __name__ == '__main__':
    main()
