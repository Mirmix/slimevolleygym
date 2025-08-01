"""
NEAT Visualization Tools

This module provides visualization tools for:
- Network topology visualization
- Evolution progress plots
- Species diversity visualization
- Fitness landscapes
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import os
from .neat_core import Genome, NEATConfig
from .neat_network import NEATNetwork

class NEATVisualizer:
    """Visualization tools for NEAT networks and evolution"""
    
    def __init__(self, output_dir: str = "neat_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_network(self, genome: Genome, config: NEATConfig, 
                         filename: str = None, show_weights: bool = True):
        """Visualize a neural network topology"""
        network = NEATNetwork(genome, config)
        
        G = nx.DiGraph()
        
        # Add nodes with colors based on type
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node_id, node_data in network.nodes.items():
            node_type = node_data['type']
            
            if node_type == 'input':
                color = '#ff9999'  # Light red
                size = 800
                label = f'I{node_id}' if node_id >= 0 else 'Bias'
            elif node_type == 'output':
                color = '#99ff99'  # Light green
                size = 800
                label = f'O{node_id}'
            elif node_type == 'bias':
                color = '#ffff99'  # Light yellow
                size = 600
                label = 'Bias'
            else:
                color = '#cccccc'  # Light gray
                size = 600
                label = f'H{node_id}'
            
            G.add_node(node_id, color=color, size=size, label=label)
            node_colors.append(color)
            node_sizes.append(size)
            node_labels[node_id] = label
        
        # Add edges
        edge_colors = []
        edge_weights = []
        
        for gene in genome.genes:
            if gene.enabled:
                G.add_edge(gene.from_node, gene.to_node, weight=gene.weight)
                edge_colors.append('red' if gene.weight < 0 else 'blue')
                edge_weights.append(abs(gene.weight))
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Use hierarchical layout for feedforward networks
        pos = self._hierarchical_layout(network)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)
        
        # Draw edges with varying thickness
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [w / max_weight * 3 + 0.5 for w in edge_weights]
        else:
            edge_widths = [1.0] * len(edge_colors)
        
        nx.draw_networkx_edges(G, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, node_labels, font_size=10, font_weight='bold')
        
        # Add title with network info (no fitness in self-play mode)
        network_info = network.get_network_info()
        title = f"NEAT Network\n"
        title += f"Nodes: {network_info['total_nodes']}, "
        title += f"Connections: {network_info['enabled_connections']}"
        
        # Add layer information if verbose layers is enabled
        if config.verbose_layers:
            input_nodes = set(range(config.input_size)) | {-1}
            output_nodes = set(range(config.input_size, config.input_size + config.output_size))
            all_nodes = set(network.nodes.keys())
            hidden_nodes = all_nodes - input_nodes - output_nodes
            layers = network._determine_network_layers(input_nodes, hidden_nodes, output_nodes)
            
            title += f"\nLayers: {len(layers)}"
            for i, layer in enumerate(layers):
                layer_type = "Input" if i == 0 else "Output" if i == len(layers) - 1 else f"Hidden {i}"
                title += f"\n  {layer_type}: {sorted(layer)}"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if filename is None:
            filename = os.path.join(self.output_dir, f'network_gen_{genome.fitness:.3f}.png')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Network visualization saved to {filename}")
    
    def _hierarchical_layout(self, network: NEATNetwork) -> Dict[int, Tuple[float, float]]:
        """Create hierarchical layout for feedforward network using actual layer structure"""
        pos = {}
        
        # Get actual layers using the network's layer determination
        input_nodes = set(range(network.config.input_size)) | {-1}  # Include bias
        output_nodes = set(range(network.config.input_size, network.config.input_size + network.config.output_size))
        all_nodes = set(network.nodes.keys())
        hidden_nodes = all_nodes - input_nodes - output_nodes
        
        # Determine actual layers using the network's method
        layers = network._determine_network_layers(input_nodes, hidden_nodes, output_nodes)
        
        # Position nodes based on actual layer structure
        for layer_idx, layer in enumerate(layers):
            layer_nodes = sorted(layer)
            
            # Calculate x position based on layer index
            x_pos = layer_idx * 4  # 4 units between layers
            
            # Calculate y positions for nodes in this layer
            if len(layer_nodes) == 1:
                y_pos = 0
            else:
                y_spacing = max(4, len(layer_nodes) * 2)
                for i, node_id in enumerate(layer_nodes):
                    y_pos = i * y_spacing / len(layer_nodes) - y_spacing / 2
                    pos[node_id] = (x_pos, y_pos)
                continue
            
            # Single node case
            pos[layer_nodes[0]] = (x_pos, 0)
        
        return pos
    
    def plot_evolution_progress(self, fitness_history: List[float], 
                               species_history: List[int],
                               complexity_history: List[float],
                               filename: str = None):
        """Plot evolution progress over generations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        generations = list(range(1, len(fitness_history) + 1))
        
        # Fitness progress
        ax1.plot(generations, fitness_history, 'b-', linewidth=2, label='Best Fitness')
        ax1.fill_between(generations, fitness_history, alpha=0.3, color='blue')
        ax1.set_title('Best Fitness Over Generations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Species count
        ax2.plot(generations, species_history, 'g-', linewidth=2, label='Species Count')
        ax2.fill_between(generations, species_history, alpha=0.3, color='green')
        ax2.set_title('Species Diversity Over Generations', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Number of Species')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Network complexity
        ax3.plot(generations, complexity_history, 'r-', linewidth=2, label='Avg Complexity')
        ax3.fill_between(generations, complexity_history, alpha=0.3, color='red')
        ax3.set_title('Average Network Complexity', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Number of Connections')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Fitness distribution (if we have current population data)
        if len(fitness_history) > 0:
            # Create a histogram of recent fitness values
            recent_fitness = fitness_history[-min(50, len(fitness_history)):]
            ax4.hist(recent_fitness, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_title('Recent Fitness Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Fitness')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            filename = os.path.join(self.output_dir, 'evolution_progress.png')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evolution progress plot saved to {filename}")
    
    def plot_species_diversity(self, species_data: List[Dict], filename: str = None):
        """Plot species diversity and fitness distribution"""
        if not species_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Species size distribution
        species_sizes = [len(species['members']) for species in species_data]
        species_fitnesses = [species['avg_fitness'] for species in species_data]
        
        ax1.scatter(species_sizes, species_fitnesses, alpha=0.7, s=100)
        ax1.set_xlabel('Species Size')
        ax1.set_ylabel('Average Fitness')
        ax1.set_title('Species Size vs Average Fitness', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Species size histogram
        ax2.hist(species_sizes, bins=min(10, len(species_sizes)), alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Species Size')
        ax2.set_ylabel('Number of Species')
        ax2.set_title('Species Size Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            filename = os.path.join(self.output_dir, 'species_diversity.png')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Species diversity plot saved to {filename}")
    
    def plot_fitness_landscape(self, population: List[Genome], filename: str = None):
        """Plot fitness landscape showing complexity vs fitness"""
        if not population:
            return
        
        complexities = [len(genome.genes) for genome in population]
        fitnesses = [genome.fitness for genome in population]
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        scatter = plt.scatter(complexities, fitnesses, 
                            c=fitnesses, cmap='viridis', 
                            alpha=0.7, s=50)
        
        plt.colorbar(scatter, label='Fitness')
        plt.xlabel('Network Complexity (Number of Connections)')
        plt.ylabel('Fitness')
        plt.title('Fitness Landscape: Complexity vs Fitness', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(complexities) > 1:
            z = np.polyfit(complexities, fitnesses, 1)
            p = np.poly1d(z)
            plt.plot(complexities, p(complexities), "r--", alpha=0.8, linewidth=2)
        
        if filename is None:
            filename = os.path.join(self.output_dir, 'fitness_landscape.png')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Fitness landscape plot saved to {filename}") 