"""
NEAT Network Implementation

This module converts NEAT genomes into feedforward neural networks
that can be used for evaluation in the Slime Volleyball environment.
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from .neat_core import Genome, NEATConfig

class NEATNetwork:
    """Converts a NEAT genome to a feedforward neural network"""
    def __init__(self, genome: Genome, config: NEATConfig):
        self.genome = genome
        self.config = config
        self.nodes = self._build_nodes()
        self.connections = self._build_connections()
        self.node_order = self._topological_sort()
    
    def _build_nodes(self) -> Dict[int, Dict]:
        """Build node dictionary from genome"""
        nodes = {}
        
        # Add input nodes
        for i in range(self.config.input_size):
            nodes[i] = {'type': 'input', 'value': 0.0}
        
        # Add bias node
        if self.config.bias_node:
            nodes[-1] = {'type': 'bias', 'value': 1.0}
        
        # Add output nodes
        for i in range(self.config.input_size, self.config.input_size + self.config.output_size):
            nodes[i] = {'type': 'output', 'value': 0.0}
        
        # Add hidden nodes from all genes (both from_node and to_node)
        for gene in self.genome.genes:
            if gene.enabled:
                # Add to_node if it's a hidden node
                if (gene.to_node >= self.config.input_size + self.config.output_size and 
                    gene.to_node not in nodes):
                    nodes[gene.to_node] = {'type': 'hidden', 'value': 0.0}
                
                # Add from_node if it's a hidden node
                if (gene.from_node >= self.config.input_size + self.config.output_size and 
                    gene.from_node not in nodes):
                    nodes[gene.from_node] = {'type': 'hidden', 'value': 0.0}
        
        return nodes
    
    def _build_connections(self) -> Dict[int, List[Tuple[int, float]]]:
        """Build connection dictionary from genome"""
        connections = {}
        
        for gene in self.genome.genes:
            if gene.enabled:
                if gene.to_node not in connections:
                    connections[gene.to_node] = []
                connections[gene.to_node].append((gene.from_node, gene.weight))
        
        return connections
    
    def _topological_sort(self) -> List[int]:
        """Create evaluation order ensuring feedforward structure"""
        # Group nodes by type
        input_nodes = []
        hidden_nodes = []
        output_nodes = []
        bias_nodes = []
        
        for node_id, node_data in self.nodes.items():
            if node_data['type'] == 'input':
                input_nodes.append(node_id)
            elif node_data['type'] == 'output':
                output_nodes.append(node_id)
            elif node_data['type'] == 'bias':
                bias_nodes.append(node_id)
            else:
                hidden_nodes.append(node_id)
        
        # Determine actual layers using topological analysis
        input_layer = set(input_nodes) | set(bias_nodes)
        hidden_layer = set(hidden_nodes)
        output_layer = set(output_nodes)
        
        layers = self._determine_network_layers(input_layer, hidden_layer, output_layer)
        
        # Create evaluation order based on actual layers
        result = []
        for layer in layers:
            result.extend(sorted(layer))
        
        return result
    
    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """Activate the network with given inputs"""
        # Reset node values
        for node_id, node_data in self.nodes.items():
            if node_data['type'] == 'input':
                if node_id < len(inputs):
                    node_data['value'] = inputs[node_id]
            elif node_data['type'] == 'bias':
                node_data['value'] = 1.0
            else:
                node_data['value'] = 0.0
        
        # Process nodes in topological order
        for node_id in self.node_order:
            if node_id in self.connections and node_id in self.nodes:
                # Sum weighted inputs
                total_input = 0.0
                for from_node, weight in self.connections[node_id]:
                    if from_node in self.nodes:
                        total_input += self.nodes[from_node]['value'] * weight
                
                # Apply activation function
                if self.nodes[node_id]['type'] != 'input' and self.nodes[node_id]['type'] != 'bias':
                    self.nodes[node_id]['value'] = self.config.activation_function(total_input)
        
        # Extract output values
        outputs = []
        for i in range(self.config.input_size, self.config.input_size + self.config.output_size):
            if i in self.nodes:
                outputs.append(self.nodes[i]['value'])
            else:
                outputs.append(0.0)
        
        return np.array(outputs)
    
    def get_network_info(self) -> Dict:
        """Get information about the network topology"""
        input_nodes = [n for n, data in self.nodes.items() if data['type'] == 'input']
        hidden_nodes = [n for n, data in self.nodes.items() if data['type'] == 'hidden']
        output_nodes = [n for n, data in self.nodes.items() if data['type'] == 'output']
        bias_nodes = [n for n, data in self.nodes.items() if data['type'] == 'bias']
        
        enabled_connections = [g for g in self.genome.genes if g.enabled]
        disabled_connections = [g for g in self.genome.genes if not g.enabled]
        
        return {
            'input_nodes': len(input_nodes),
            'hidden_nodes': len(hidden_nodes),
            'output_nodes': len(output_nodes),
            'bias_nodes': len(bias_nodes),
            'total_nodes': len(self.nodes),
            'enabled_connections': len(enabled_connections),
            'disabled_connections': len(disabled_connections),
            'total_connections': len(self.genome.genes),
            'fitness': self.genome.fitness
        }
    
    def save_network(self, filename: str):
        """Save network to file"""
        network_data = {
            'genome_genes': [(g.innovation_number, g.from_node, g.to_node, 
                             g.weight, g.enabled) for g in self.genome.genes],
            'config': {
                'input_size': self.config.input_size,
                'output_size': self.config.output_size,
                'bias_node': self.config.bias_node,
                'activation_function': 'tanh'
            },
            'fitness': self.genome.fitness,
            'network_info': self.get_network_info()
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2)
    
    @classmethod
    def load_network(cls, filename: str, config: NEATConfig):
        """Load network from file"""
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Recreate genome
        genome = Genome(config)
        genome.genes = []
        
        for gene_data in data['genome_genes']:
            innovation, from_node, to_node, weight, enabled = gene_data
            from .neat_core import Gene
            gene = Gene(innovation, from_node, to_node, weight, enabled)
            genome.genes.append(gene)
        
        genome.fitness = data['fitness']
        
        return cls(genome, config) 

    def validate_feedforward(self) -> bool:
        """Validate that the network is purely feedforward"""
        if not self.config.validate_networks:
            return True  # Skip validation if disabled
        
        # Get all nodes and their layers using proper topological analysis
        input_nodes = set(range(self.config.input_size)) | {-1}  # Include bias
        output_nodes = set(range(self.config.input_size, self.config.input_size + self.config.output_size))
        
        # Get hidden nodes
        all_nodes = set(self.nodes.keys())
        hidden_nodes = all_nodes - input_nodes - output_nodes
        
        # Determine actual layers using topological analysis
        layers = self._determine_network_layers(input_nodes, hidden_nodes, output_nodes)
        
        # Check all connections for feedforward validity
        for gene in self.genome.genes:
            if not gene.enabled:
                continue
                
            from_node = gene.from_node
            to_node = gene.to_node
            
            # Check for self-connections
            if from_node == to_node:
                return False
            
            # Check for same-layer connections
            from_layer_idx = self._get_node_layer_index(from_node, layers)
            to_layer_idx = self._get_node_layer_index(to_node, layers)
            
            if from_layer_idx == to_layer_idx:
                return False
            
            # Check for backward connections
            if from_layer_idx >= to_layer_idx:
                return False
        
        return True
    
    def _determine_network_layers(self, input_nodes: set, hidden_nodes: set, output_nodes: set) -> list:
        """Determine actual network layers using topological analysis for arbitrary multi-layer networks"""
        # Start with input layer
        layers = [input_nodes]
        
        # Build adjacency list for hidden nodes
        adjacency = {node: set() for node in hidden_nodes}
        for gene in self.genome.genes:
            if gene.enabled and gene.from_node in hidden_nodes and gene.to_node in hidden_nodes:
                adjacency[gene.from_node].add(gene.to_node)
        
        # Check if this is a single hidden layer (all nodes are interconnected)
        if hidden_nodes:
            # Create a graph to detect if all hidden nodes are in the same component
            visited = set()
            components = []
            
            for node in hidden_nodes:
                if node not in visited:
                    # DFS to find connected component
                    component = set()
                    stack = [node]
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            component.add(current)
                            # Add all neighbors
                            for gene in self.genome.genes:
                                if gene.enabled:
                                    if gene.from_node == current and gene.to_node in hidden_nodes:
                                        stack.append(gene.to_node)
                                    elif gene.to_node == current and gene.from_node in hidden_nodes:
                                        stack.append(gene.from_node)
                    components.append(component)
            
            # If all hidden nodes are in one component, treat as single layer
            if len(components) == 1 and len(components[0]) == len(hidden_nodes):
                # Single hidden layer - all nodes are interconnected
                layers.append(hidden_nodes)
            else:
                # Multiple hidden layers - use topological sort
                # Find nodes with no incoming connections from other hidden nodes (first hidden layer)
                current_layer = set()
                for node in hidden_nodes:
                    has_incoming_from_hidden = False
                    for gene in self.genome.genes:
                        if gene.enabled and gene.to_node == node and gene.from_node in hidden_nodes:
                            has_incoming_from_hidden = True
                            break
                    if not has_incoming_from_hidden:
                        current_layer.add(node)
                
                # Build layers using topological sort for arbitrary depth
                remaining_nodes = hidden_nodes - current_layer
                layer_count = 1
                
                while current_layer:
                    layers.append(current_layer)
                    next_layer = set()
                    
                    # Find nodes that can be reached from current layer
                    for node in remaining_nodes:
                        can_reach = False
                        for gene in self.genome.genes:
                            if gene.enabled and gene.from_node in current_layer and gene.to_node == node:
                                can_reach = True
                                break
                        if can_reach:
                            next_layer.add(node)
                    
                    current_layer = next_layer
                    remaining_nodes -= next_layer
                    layer_count += 1
                    
                    # Safety check to prevent infinite loops
                    if layer_count > 100:  # Arbitrary limit
                        break
                
                # Handle any remaining nodes (isolated or in cycles) by placing them in a new layer
                if remaining_nodes:
                    layers.append(remaining_nodes)
        
        # Add output layer
        layers.append(output_nodes)
        
        return layers
    
    def _get_node_layer_index(self, node: int, layers: list) -> int:
        """Get the layer index of a node based on actual network structure"""
        for i, layer in enumerate(layers):
            if node in layer:
                return i
        # If not found, assume it's in the last layer (output)
        return len(layers) - 1 

    def get_network_architecture(self) -> Dict:
        """Get detailed information about the network architecture"""
        input_nodes = set(range(self.config.input_size)) | {-1}  # Include bias
        output_nodes = set(range(self.config.input_size, self.config.input_size + self.config.output_size))
        all_nodes = set(self.nodes.keys())
        hidden_nodes = all_nodes - input_nodes - output_nodes
        
        # Determine actual layers
        layers = self._determine_network_layers(input_nodes, hidden_nodes, output_nodes)
        
        # Analyze connections between layers
        layer_connections = {}
        for i in range(len(layers) - 1):
            from_layer = layers[i]
            to_layer = layers[i + 1]
            connections = 0
            for gene in self.genome.genes:
                if gene.enabled and gene.from_node in from_layer and gene.to_node in to_layer:
                    connections += 1
            layer_connections[f"Layer_{i}_to_{i+1}"] = connections
        
        # Count connections within layers (should be 0 for feedforward)
        same_layer_connections = 0
        for layer in layers:
            for gene in self.genome.genes:
                if gene.enabled and gene.from_node in layer and gene.to_node in layer:
                    same_layer_connections += 1
        
        return {
            'total_layers': len(layers),
            'layer_sizes': [len(layer) for layer in layers],
            'layer_types': ['Input' if i == 0 else 'Output' if i == len(layers) - 1 else f'Hidden_{i}' for i in range(len(layers))],
            'layer_connections': layer_connections,
            'same_layer_connections': same_layer_connections,
            'is_feedforward': same_layer_connections == 0,
            'total_enabled_connections': sum(1 for g in self.genome.genes if g.enabled),
            'total_disabled_connections': sum(1 for g in self.genome.genes if not g.enabled)
        }
    
    def print_network_architecture(self):
        """Print detailed network architecture information"""
        arch = self.get_network_architecture()
        
        print(f"\n=== Network Architecture Analysis ===")
        print(f"Total layers: {arch['total_layers']}")
        print(f"Layer sizes: {arch['layer_sizes']}")
        print(f"Layer types: {arch['layer_types']}")
        print(f"Same-layer connections: {arch['same_layer_connections']}")
        print(f"Feedforward: {'✅ Yes' if arch['is_feedforward'] else '❌ No'}")
        print(f"Enabled connections: {arch['total_enabled_connections']}")
        print(f"Disabled connections: {arch['total_disabled_connections']}")
        
        print(f"\nLayer connections:")
        for connection, count in arch['layer_connections'].items():
            print(f"  {connection}: {count} connections")
        
        # Show layer details
        input_nodes = set(range(self.config.input_size)) | {-1}
        output_nodes = set(range(self.config.input_size, self.config.input_size + self.config.output_size))
        all_nodes = set(self.nodes.keys())
        hidden_nodes = all_nodes - input_nodes - output_nodes
        layers = self._determine_network_layers(input_nodes, hidden_nodes, output_nodes)
        
        print(f"\nLayer details:")
        for i, layer in enumerate(layers):
            layer_type = "Input" if i == 0 else "Output" if i == len(layers) - 1 else f"Hidden {i}"
            print(f"  Layer {i} ({layer_type}): {sorted(layer)}") 