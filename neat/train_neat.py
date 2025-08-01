"""
NEAT Training Script for Slime Volleyball

This script implements the complete NEAT training pipeline:
- Population initialization and evolution
- Fitness evaluation using the Slime Volleyball environment
- Visualization of network topologies and evolution progress
- Saving and loading of trained agents
- GIF generation of best agents in action
"""

import os
import time
import json
import numpy as np
import gym
import slimevolleygym
from slimevolleygym import multiagent_rollout as rollout
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from .neat_core import NEATConfig, Genome, InnovationTracker
from .neat_network import NEATNetwork
from .neat_population import NEATPopulation
from .neat_visualization import NEATVisualizer

class NEATTrainer:
    """Main trainer class for NEAT algorithm"""
    def __init__(self, config: NEATConfig):
        self.config = config
        self.population = NEATPopulation(config)
        self.env = gym.make("SlimeVolley-v0")
        self.env.seed(42)
        
        # Statistics tracking
        self.fitness_history = []
        self.species_history = []
        self.complexity_history = []
        self.best_genomes_history = []
        
        # Create output directory (will be set in train method)
        self.output_dir = None
        
        # Initialize visualizer (will be set in train method)
        self.visualizer = None
        
        # Self-play tracking
        self.previous_best_genome = None
        
        # Best genome tracking for final save
        self.last_best_genome = None
        self.last_best_fitness = 0.0
        
        # All-time best genome tracking
        self.best_genome_all_time = None
        self.best_fitness_all_time = 0.0
        
        print(f"NEAT Trainer initialized with population size: {config.population_size}")
        print(f"Species threshold: {config.species_threshold}")
    
    def setup_output_directory(self, use_selfplay: bool = False):
        """Set up output directory based on training mode"""
        if use_selfplay:
            self.output_dir = "neat_results_selfplay"
        else:
            self.output_dir = "neat_results_baseline"
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer = NEATVisualizer(self.output_dir)
        
        print(f"Output directory: {self.output_dir}")
    
    def evaluate_genome(self, genome: Genome, num_episodes: int = 5) -> float:
        """Evaluate a genome's fitness by playing games against the baseline agent"""
        network = NEATNetwork(genome, self.config)
        
        # Validate feedforward structure
        if self.config.validate_networks and not network.validate_feedforward():
            return -1000.0  # Heavy penalty for non-feedforward networks
        
        total_fitness = 0.0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            steps = 0
            
            while steps < 3000:  # Max steps per episode
                # Get network output
                action_output = network.activate(obs)
                
                # Convert to action (3 binary actions: forward, backward, jump)
                action = np.array([
                    1 if action_output[0] > 0 else 0,  # forward
                    1 if action_output[1] > 0 else 0,  # backward
                    1 if action_output[2] > 0 else 0   # jump
                ])
                
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            total_fitness += episode_reward
        
        # Apply complexity penalty to prevent bloat
        raw_fitness = total_fitness / num_episodes
        complexity_penalty = self._calculate_complexity_penalty(genome)
        adjusted_fitness = raw_fitness - complexity_penalty
        
        return adjusted_fitness
    
    def _calculate_complexity_penalty(self, genome: Genome) -> float:
        """Calculate complexity penalty to prevent network bloat"""
        if not self.config.complexity_penalty_enabled:
            return 0.0
        
        # Count enabled connections (actual complexity)
        enabled_connections = sum(1 for g in genome.genes if g.enabled)
        
        # Base penalty: small penalty for each connection beyond minimal topology
        minimal_connections = self.config.input_size * self.config.output_size
        excess_connections = max(0, enabled_connections - minimal_connections)
        
        # Complexity penalty using config parameters
        penalty = self.config.complexity_coefficient * excess_connections
        
        # Additional penalty for very large networks (exponential penalty)
        if enabled_connections > self.config.complexity_threshold:
            penalty += self.config.complexity_exponential_rate * (enabled_connections - self.config.complexity_threshold)
        
        return penalty
    
    def evaluate_genome_selfplay(self, genome: Genome, opponent: Genome, 
                                num_episodes: int = 3) -> float:
        """Evaluate a genome through self-play against another genome"""
        network1 = NEATNetwork(genome, self.config)
        network2 = NEATNetwork(opponent, self.config)
        
        # Validate feedforward structure for both networks
        if self.config.validate_networks and not network1.validate_feedforward():
            return -1000.0  # Heavy penalty for non-feedforward networks
        
        total_fitness = 0.0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            steps = 0
            
            while steps < 3000:
                # Get actions from both networks
                action1_output = network1.activate(obs)
                action2_output = network2.activate(obs)
                
                action1 = np.array([
                    1 if action1_output[0] > 0 else 0,
                    1 if action1_output[1] > 0 else 0,
                    1 if action1_output[2] > 0 else 0
                ])
                
                action2 = np.array([
                    1 if action2_output[0] > 0 else 0,
                    1 if action2_output[1] > 0 else 0,
                    1 if action2_output[2] > 0 else 0
                ])
                
                obs, reward, done, info = self.env.step(action1, action2)
                episode_reward += reward  # This is the reward for agent1
                steps += 1
                
                if done:
                    break
            
            total_fitness += episode_reward
        
        # Apply complexity penalty to prevent bloat
        raw_fitness = total_fitness / num_episodes
        complexity_penalty = self._calculate_complexity_penalty(genome)
        adjusted_fitness = raw_fitness - complexity_penalty
        
        return adjusted_fitness
    
    def evaluate_population(self, use_selfplay: bool = False):
        """Evaluate all genomes in the population"""
        print(f"Evaluating generation {self.population.generation}...")
        
        if use_selfplay:
            # Self-play evaluation: use the best genome from previous generation as opponent
            if hasattr(self, 'previous_best_genome') and self.previous_best_genome is not None:
                opponent = self.previous_best_genome
                print(f"  Self-play: All genomes competing against best from generation {self.population.generation - 1}")
            else:
                # For first generation, use a random genome as opponent
                opponent = np.random.choice(self.population.population)
                print(f"  Self-play: First generation using random opponent")
            
            # Evaluate all genomes against the same opponent for fair comparison
            for i, genome in enumerate(self.population.population):
                genome.fitness = self.evaluate_genome_selfplay(genome, opponent)
                
                if i % 10 == 0:
                    enabled_connections = sum(1 for g in genome.genes if g.enabled)
                    penalty = self._calculate_complexity_penalty(genome)
                    print(f"  Genome {i+1}/{len(self.population.population)}: {genome.fitness:.3f} (connections: {enabled_connections}, penalty: {penalty:.3f})")
        else:
            # Standard evaluation against baseline
            for i, genome in enumerate(self.population.population):
                genome.fitness = self.evaluate_genome(genome)
                
                if i % 10 == 0:
                    enabled_connections = sum(1 for g in genome.genes if g.enabled)
                    penalty = self._calculate_complexity_penalty(genome)
                    print(f"  Genome {i+1}/{len(self.population.population)}: {genome.fitness:.3f} (connections: {enabled_connections}, penalty: {penalty:.3f})")
    
    def create_agent_gif(self, genome: Genome, filename: str, duration: int = None):
        """Create a GIF of the agent playing"""
        if duration is None:
            duration = self.config.gif_duration
        
        network = NEATNetwork(genome, self.config)
        
        frames = []
        obs = self.env.reset()
        
        # Ensure environment is in the right mode for rendering
        self.env.render(mode='human')
        
        for step in range(duration * self.config.gif_fps):  # Use config FPS
            # Get network output
            action_output = network.activate(obs)
            
            # Convert to action
            action = np.array([
                1 if action_output[0] > 0 else 0,
                1 if action_output[1] > 0 else 0,
                1 if action_output[2] > 0 else 0
            ])
            
            obs, reward, done, info = self.env.step(action)
            
            # Render frame
            try:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                pass  # Silently ignore rendering issues
            
            if done:
                break
        
        # Save as GIF
        if frames:
            try:
                import imageio
                # Calculate frame duration based on config FPS
                frame_duration = 1000 // self.config.gif_fps  # milliseconds per frame
                imageio.mimsave(filename, frames, duration=frame_duration)
                print(f"Saved agent GIF to {filename} ({len(frames)} frames, {duration}s duration, {self.config.gif_fps} FPS)")
                return True
            except ImportError:
                print("imageio not available, trying alternative method...")
                try:
                    # Alternative: save as individual frames
                    import matplotlib.pyplot as plt
                    import matplotlib.image as mpimg
                    
                    for i, frame in enumerate(frames):
                        plt.imsave(f"{filename.replace('.gif', '')}_frame_{i:03d}.png", frame)
                    
                    print(f"Saved individual frames instead of GIF")
                    return True
                except Exception as e:
                    print(f"Could not save frames: {e}")
                    return False
            except Exception as e:
                print(f"Error saving GIF: {e}")
                return False
        else:
            print("No frames captured, skipping GIF creation")
            return False
    
    def save_best_agent(self, create_gif=True):
        """Save the best agent for later use"""
        if self.population.best_genome:
            best_network = NEATNetwork(self.population.best_genome, self.config)
            
            # Save as JSON
            agent_data = {
                'fitness': self.population.best_genome.fitness,
                'generation': self.population.generation,
                'genes': [(g.innovation_number, g.from_node, g.to_node, 
                          g.weight, g.enabled) for g in self.population.best_genome.genes],
                'config': {
                    'input_size': self.config.input_size,
                    'output_size': self.config.output_size,
                    'activation_function': 'tanh'
                }
            }
            
            with open(os.path.join(self.output_dir, 'best_agent.json'), 'w') as f:
                json.dump(agent_data, f, indent=2)
            
            print(f"  💾 SAVED: best_agent.json with fitness: {self.population.best_genome.fitness:.3f}")
            print(f"  📊 Genes: {len(self.population.best_genome.genes)}")
            print(f"  🧠 Enabled connections: {sum(1 for g in self.population.best_genome.genes if g.enabled)}")
            
            # Create GIF of best agent (if requested)
            if create_gif:
                gif_success = self.create_agent_gif(
                    self.population.best_genome, 
                    os.path.join(self.output_dir, 'best_agent.gif')
                )
                
                if not gif_success:
                    print("Note: GIF creation failed. Check if imageio is installed or try running with --no-gif flag")
    
    def train(self, max_generations: int = 100, use_selfplay: bool = False, 
              save_frequency: int = 10, create_gif: bool = True):
        """Train the NEAT population"""
        # Set up output directory based on training mode
        self.setup_output_directory(use_selfplay)
        
        print(f"Starting NEAT training for {max_generations} generations...")
        print(f"Population size: {self.config.population_size}")
        print(f"Species threshold: {self.config.species_threshold}")
        print(f"Self-play mode: {use_selfplay}")
        
        # Clean up any existing non-feedforward connections in the population
        if self.config.cleanup_non_feedforward:
            print("Cleaning up non-feedforward connections in initial population...")
            for genome in self.population.population:
                genome.cleanup_non_feedforward_connections()
        
        start_time = time.time()
        
        for generation in range(max_generations):
            print(f"\n=== Generation {generation + 1} ===")
            
            # Evaluate population
            self.evaluate_population(use_selfplay=use_selfplay)
            
            # Update best genome immediately after evaluation
            current_best_fitness = max(g.fitness for g in self.population.population)
            current_best_genome = max(self.population.population, key=lambda g: g.fitness)
            
            # Handle case where all fitness values are the same (e.g., all negative)
            if len(set(g.fitness for g in self.population.population)) == 1:
                current_best_genome = self.population.population[0]  # Just pick the first one
            
            if current_best_fitness > self.population.best_fitness:
                self.population.best_fitness = current_best_fitness
                self.population.best_genome = current_best_genome
                print(f"  New best genome found: {current_best_fitness:.3f}")
            else:
                if current_best_fitness > 0:
                    print(f"  Current best fitness: {current_best_fitness:.3f} (positive performance against stronger opponent)")
                else:
                    print(f"  Current best fitness: {current_best_fitness:.3f} (no improvement)")
            
            # Store current best for next generation's self-play
            self.previous_best_genome = current_best_genome.copy()
            
            # Record statistics
            stats = self.population.get_statistics()
            self.fitness_history.append(current_best_fitness)
            self.species_history.append(stats['species_count'])
            self.complexity_history.append(stats['average_complexity'])
            
            # Store best genome for visualization
            if self.population.best_genome:
                self.best_genomes_history.append(self.population.best_genome)
            
            # Display statistics
            print(f"Best fitness: {current_best_fitness:.3f}")
            print(f"Average fitness: {stats['average_fitness']:.3f}")
            print(f"Species count: {stats['species_count']}")
            print(f"Average complexity: {stats['average_complexity']:.1f} connections")
            
            # Save progress BEFORE evolution (save current generation's best)
            if generation % save_frequency == 0:
                # Temporarily save the current generation's best genome
                temp_best_genome = current_best_genome.copy()
                temp_best_fitness = current_best_fitness
                
                # Temporarily set population best to current generation's best
                original_best_genome = self.population.best_genome
                original_best_fitness = self.population.best_fitness
                self.population.best_genome = temp_best_genome
                self.population.best_fitness = temp_best_fitness
                
                # Save the current generation's best
                self.save_best_agent(create_gif=create_gif)
                
                self.plot_progress()
                
                # Visualize current generation's best network
                self.visualizer.visualize_network(
                    temp_best_genome,
                    self.config,
                    os.path.join(self.output_dir, f'best_network_gen_{generation}.png')
                )
                
                # Restore original best genome
                self.population.best_genome = original_best_genome
                self.population.best_fitness = original_best_fitness
            
            # Check for convergence
            if current_best_fitness >= 5.0:  # Maximum possible score
                print(f"Maximum fitness reached! Best fitness: {current_best_fitness}")
                break
            
            # For self-play: ensure the best genome is preserved for next generation
            if use_selfplay:
                # Store the best genome from current generation for next generation's self-play
                self.last_best_genome = current_best_genome.copy()
                self.last_best_fitness = current_best_fitness
                
                # Evolve population but preserve the best genome
                self.population.evolve()
                
                # Ensure the best genome from previous generation is still available for self-play
                if self.last_best_genome is not None:
                    self.previous_best_genome = self.last_best_genome
            else:
                # Standard evolution
                self.population.evolve()
            
            # Restore best genome if evolution didn't improve it
            if current_best_fitness > self.population.best_fitness:
                self.population.best_fitness = current_best_fitness
                self.population.best_genome = current_best_genome
        
        # Final save - use the last evaluated generation's best
        # Since we're in self-play mode, we want the best from the last generation
        final_best_fitness = max(g.fitness for g in self.population.population)
        final_best_genome = max(self.population.population, key=lambda g: g.fitness)
        
        # Update population best if this is better
        if final_best_fitness > self.population.best_fitness:
            self.population.best_fitness = final_best_fitness
            self.population.best_genome = final_best_genome
            print(f"  Final best genome: {final_best_fitness:.3f}")
        else:
            print(f"  Final best fitness: {final_best_fitness:.3f}")
        
        # Final save and visualization
        self.save_best_agent(create_gif=create_gif)
        self.plot_progress()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best fitness achieved: {self.population.best_fitness:.3f}")
    
    def plot_progress(self):
        """Plot evolution progress"""
        if len(self.fitness_history) > 0:
            self.visualizer.plot_evolution_progress(
                self.fitness_history,
                self.species_history,
                self.complexity_history,
                os.path.join(self.output_dir, 'evolution_progress.png')
            )

def main():
    """Main function to run NEAT training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NEAT agents for Slime Volleyball')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations to train')
    parser.add_argument('--population', type=int, default=100, help='Population size')
    parser.add_argument('--selfplay', action='store_true', help='Use self-play instead of baseline evaluation')
    parser.add_argument('--no-gif', action='store_true', help='Disable GIF creation')
    parser.add_argument('--test', action='store_true', help='Run quick test instead of full training')
    parser.add_argument('--gif-duration', type=int, default=30, help='Duration of agent GIFs in seconds')
    parser.add_argument('--gif-fps', type=int, default=30, help='Frames per second for GIFs')
    
    args = parser.parse_args()
    
    # Create configuration
    config = NEATConfig()
    
    # Adjust configuration based on arguments
    config.population_size = args.population
    config.mutate_add_connection_rate = 0.1  # Higher rate for more exploration
    config.mutate_add_node_rate = 0.05
    
    # Set GIF settings from arguments
    config.gif_duration = args.gif_duration
    config.gif_fps = args.gif_fps
    
    # Create trainer
    trainer = NEATTrainer(config)
    
    if args.test:
        # Run quick test
        print("Running quick test...")
        trainer.train(max_generations=5, use_selfplay=args.selfplay, create_gif=not args.no_gif)
    else:
        # Start training
        print("Starting NEAT training...")
        trainer.train(max_generations=args.generations, use_selfplay=args.selfplay, create_gif=not args.no_gif)
    
    print("\nTraining completed!")
    print("Check the 'neat_results' directory for outputs:")
    print("- best_agent.json: Best trained agent")
    if not args.no_gif:
        print(f"- best_agent.gif: GIF of best agent playing ({args.gif_duration}s, {args.gif_fps} FPS)")
    print("- evolution_progress.png: Training progress plots")

if __name__ == "__main__":
    main() 