"""
Script to load a trained NEAT model and play it against the baseline agent
"""

import os
import json
import numpy as np
import gym
import slimevolleygym
import matplotlib.pyplot as plt
from typing import Dict, Any

from .neat_core import NEATConfig, Genome, Gene
from .neat_network import NEATNetwork

class ModelPlayer:
    """Load and play a trained NEAT model against the baseline"""
    
    def __init__(self, model_path: str = "neat_results/best_agent.json"):
        self.model_path = model_path
        self.config = NEATConfig()
        self.env = gym.make("SlimeVolley-v0")
        self.env.seed(42)
        
        # Load the trained model
        self.genome = self.load_model(model_path)
        self.network = NEATNetwork(self.genome, self.config)
        
        print(f"Loaded model from: {model_path}")
        print(f"Model fitness: {self.genome.fitness:.3f}")
        print(f"Network complexity: {len(self.genome.genes)} connections")
    
    def load_model(self, model_path: str) -> Genome:
        """Load a trained model from JSON file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Create genome from saved data
        genome = Genome(self.config)
        genome.fitness = model_data.get('fitness', 0.0)
        
        # Load genes
        genome.genes = []
        for gene_data in model_data['genes']:
            innovation, from_node, to_node, weight, enabled = gene_data
            gene = Gene(innovation, from_node, to_node, weight, enabled)
            genome.genes.append(gene)
        
        return genome
    
    def play_episode(self, render: bool = True, max_steps: int = 3000) -> Dict[str, Any]:
        """Play a single episode against the baseline"""
        obs = self.env.reset()
        total_reward = 0.0
        steps = 0
        points_scored = 0
        points_lost = 0
        
        if render:
            self.env.render(mode='human')
        
        while steps < max_steps:
            # Get network output
            action_output = self.network.activate(obs)
            
            # Convert to action
            action = np.array([
                1 if action_output[0] > 0 else 0,  # forward
                1 if action_output[1] > 0 else 0,  # backward
                1 if action_output[2] > 0 else 0   # jump
            ])
            
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Track points
            if reward == 1:
                points_scored += 1
            elif reward == -1:
                points_lost += 1
            
            steps += 1
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'points_scored': points_scored,
            'points_lost': points_lost,
            'net_score': points_scored - points_lost,
            'win_rate': points_scored / max(1, points_scored + points_lost)
        }
    
    def play_multiple_episodes(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """Play multiple episodes and collect statistics"""
        print(f"Playing {num_episodes} episodes against baseline...")
        
        episode_results = []
        total_rewards = []
        total_points_scored = 0
        total_points_lost = 0
        
        for episode in range(num_episodes):
            result = self.play_episode(render=render)
            episode_results.append(result)
            total_rewards.append(result['total_reward'])
            total_points_scored += result['points_scored']
            total_points_lost += result['points_lost']
            
            print(f"Episode {episode + 1}: Score {result['points_scored']}-{result['points_lost']} "
                  f"(Reward: {result['total_reward']:.1f})")
        
        # Calculate statistics
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        total_win_rate = total_points_scored / max(1, total_points_scored + total_points_lost)
        
        stats = {
            'num_episodes': num_episodes,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'total_points_scored': total_points_scored,
            'total_points_lost': total_points_lost,
            'total_win_rate': total_win_rate,
            'episode_results': episode_results
        }
        
        print(f"\n=== Final Statistics ===")
        print(f"Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"Total Score: {total_points_scored}-{total_points_lost}")
        print(f"Win Rate: {total_win_rate:.1%}")
        
        return stats
    
    def analyze_performance(self, stats: Dict[str, Any]):
        """Analyze the model's performance"""
        print(f"\n=== Performance Analysis ===")
        
        # Performance rating
        avg_reward = stats['avg_reward']
        win_rate = stats['total_win_rate']
        
        if avg_reward > 2.0 and win_rate > 0.7:
            rating = "Excellent"
        elif avg_reward > 1.0 and win_rate > 0.5:
            rating = "Good"
        elif avg_reward > 0.0 and win_rate > 0.3:
            rating = "Fair"
        elif avg_reward > -1.0:
            rating = "Poor"
        else:
            rating = "Very Poor"
        
        print(f"Performance Rating: {rating}")
        print(f"Model successfully beats baseline: {'Yes' if avg_reward > 0 else 'No'}")
        
        # Compare with baseline
        if avg_reward > 0:
            print(f"Model outperforms baseline by {avg_reward:.2f} points per episode")
        else:
            print(f"Model underperforms baseline by {abs(avg_reward):.2f} points per episode")
    
    def visualize_network(self, save_path: str = "neat_results/loaded_network.png"):
        """Visualize the loaded network"""
        try:
            from .neat_visualization import NEATVisualizer
            visualizer = NEATVisualizer("neat_results")
            visualizer.visualize_network(self.genome, self.config, save_path)
            print(f"Network visualization saved to: {save_path}")
        except ImportError:
            print("Visualization module not available")
    
    def close(self):
        """Clean up resources"""
        self.env.close()

def main():
    """Main function to run the model player"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Play a trained NEAT model against baseline')
    parser.add_argument('--model', type=str, default='neat_results/best_agent.json', 
                       help='Path to the trained model JSON file')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Number of episodes to play')
    parser.add_argument('--render', action='store_true', 
                       help='Render the game (slower but visual)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create network visualization')
    
    args = parser.parse_args()
    
    try:
        # Create player
        player = ModelPlayer(args.model)
        
        # Play episodes
        stats = player.play_multiple_episodes(
            num_episodes=args.episodes, 
            render=args.render
        )
        
        # Analyze performance
        player.analyze_performance(stats)
        
        # Visualize network if requested
        if args.visualize:
            player.visualize_network()
        
        player.close()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have trained a model first using:")
        print("python -m neat.train_neat")
        
    except Exception as e:
        print(f"Error running model player: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 