"""
Generate example results and visualizations for the README
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from .neat_core import NEATConfig, Genome
from .neat_network import NEATNetwork
from .neat_visualization import NEATVisualizer

def generate_example_results():
    """Generate example results for the README"""
    
    # Create output directory
    os.makedirs("neat_results", exist_ok=True)
    
    # Create configuration
    config = NEATConfig()
    config.population_size = 100
    
    # Create example genomes with different complexities
    genomes = []
    
    # Simple genome (early generation)
    simple_genome = Genome(config)
    simple_genome.fitness = -1.5
    genomes.append(simple_genome)
    
    # Medium complexity genome (middle generation)
    medium_genome = Genome(config)
    # Add some mutations to make it more complex
    for _ in range(5):
        medium_genome.mutate(medium_genome.config.innovation_tracker)
    medium_genome.fitness = 1.2
    genomes.append(medium_genome)
    
    # Complex genome (late generation)
    complex_genome = Genome(config)
    # Add more mutations
    for _ in range(15):
        complex_genome.mutate(complex_genome.config.innovation_tracker)
    complex_genome.fitness = 3.2
    genomes.append(complex_genome)
    
    # Create visualizer
    visualizer = NEATVisualizer("neat_results")
    
    # Generate network visualizations
    print("Generating network visualizations...")
    visualizer.visualize_network(simple_genome, config, "simple_network.png")
    visualizer.visualize_network(medium_genome, config, "medium_network.png")
    visualizer.visualize_network(complex_genome, config, "complex_network.png")
    
    # Generate evolution progress plot
    print("Generating evolution progress plot...")
    fitness_history = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.2]
    species_history = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10]
    complexity_history = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    visualizer.plot_evolution_progress(fitness_history, species_history, complexity_history)
    
    # Create example agent data
    print("Creating example agent data...")
    best_agent_data = {
        'fitness': 3.2,
        'generation': 100,
        'genes': [(g.innovation_number, g.from_node, g.to_node, g.weight, g.enabled) 
                 for g in complex_genome.genes],
        'config': {
            'input_size': config.input_size,
            'output_size': config.output_size,
            'activation_function': 'tanh'
        }
    }
    
    import json
    with open("neat_results/best_agent.json", 'w') as f:
        json.dump(best_agent_data, f, indent=2)
    
    print("Example results generated in neat_results/ directory!")
    print("Files created:")
    print("- simple_network.png")
    print("- medium_network.png") 
    print("- complex_network.png")
    print("- evolution_progress.png")
    print("- best_agent.json")

if __name__ == "__main__":
    generate_example_results() 