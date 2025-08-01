"""
NEAT (NeuroEvolution of Augmenting Topologies) Core Implementation

This module contains the core NEAT components:
- Gene and Genome classes
- Innovation tracking
- Mutation operators
- Crossover operators
- Species management
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set
import json

class NEATConfig:
    """Configuration for NEAT algorithm"""
    def __init__(self):
        # Population settings
        self.population_size = 150
        self.species_threshold = 1.0
        self.species_elitism = 2
        
        # Mutation rates
        self.mutate_connection_weight_rate = 0.8
        self.mutate_add_connection_rate = 0.15
        self.mutate_add_node_rate = 0.08
        self.mutate_toggle_connection_rate = 0.1
        
        # Mutation parameters
        self.weight_mutation_power = 0.1
        self.weight_mutation_rate = 0.9
        self.weight_replace_rate = 0.1
        
        # Crossover settings
        self.crossover_rate = 0.75
        self.disjoint_coefficient = 1.0
        self.excess_coefficient = 1.0
        self.weight_coefficient = 0.4
        
        # Network settings
        self.input_size = 12  # Slime volleyball state size
        self.output_size = 3   # Slime volleyball action size
        self.bias_node = True
        self.activation_function = np.tanh
        
        # Complexity penalization settings
        self.complexity_penalty_enabled = True
        self.complexity_coefficient = 0.01  # Penalty per excess connection
        self.complexity_threshold = 100     # Threshold for exponential penalty
        self.complexity_exponential_rate = 0.1  # Rate for exponential penalty
        
        # Feedforward architecture settings
        self.enforce_feedforward = True     # Enforce purely feedforward networks
        self.validate_networks = True       # Validate networks during evaluation
        self.cleanup_non_feedforward = True # Clean up non-feedforward connections
        self.verbose_cleanup = False        # Print detailed cleanup information
        self.verbose_layers = False         # Print layer information for debugging
        
        # GIF and visualization settings
        self.gif_duration = 30              # Duration of agent GIFs in seconds
        self.gif_fps = 30                   # Frames per second for GIFs

class Gene:
    """Represents a single gene in the NEAT genome"""
    def __init__(self, innovation_number: int, from_node: int, to_node: int, 
                 weight: float, enabled: bool = True):
        self.innovation_number = innovation_number
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled
    
    def copy(self):
        return Gene(self.innovation_number, self.from_node, self.to_node, 
                   self.weight, self.enabled)
    
    def mutate_weight(self, power: float, rate: float, replace_rate: float):
        if random.random() < rate:
            if random.random() < replace_rate:
                self.weight = random.gauss(0, power)
            else:
                self.weight += random.gauss(0, power)
    
    def __eq__(self, other):
        return (self.from_node == other.from_node and 
                self.to_node == other.to_node)
    
    def __hash__(self):
        return hash((self.from_node, self.to_node))

class Genome:
    """Represents a complete NEAT genome"""
    def __init__(self, config: NEATConfig):
        self.config = config
        self.genes: List[Gene] = []
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = None
        
        # Initialize with minimal topology
        self._initialize_minimal_topology()
    
    def _initialize_minimal_topology(self):
        """Initialize genome with minimal topology (input -> output)"""
        input_nodes = list(range(self.config.input_size))
        if self.config.bias_node:
            input_nodes.append(-1)  # Bias node
        
        output_nodes = list(range(self.config.input_size, 
                                 self.config.input_size + self.config.output_size))
        
        # Add connections from each input to each output
        innovation = 0
        for from_node in input_nodes:
            for to_node in output_nodes:
                weight = random.gauss(0, 1.0)
                self.genes.append(Gene(innovation, from_node, to_node, weight))
                innovation += 1
    
    def get_nodes(self) -> Set[int]:
        """Get all node IDs in the genome"""
        nodes = set()
        for gene in self.genes:
            nodes.add(gene.from_node)
            nodes.add(gene.to_node)
        return nodes
    
    def mutate(self, innovation_tracker):
        """Mutate the genome"""
        # Mutate connection weights
        for gene in self.genes:
            gene.mutate_weight(self.config.weight_mutation_power,
                             self.config.weight_mutation_rate,
                             self.config.weight_replace_rate)
        
        # Add new connection
        if random.random() < self.config.mutate_add_connection_rate:
            self._add_connection(innovation_tracker)
        
        # Add new node
        if random.random() < self.config.mutate_add_node_rate:
            self._add_node(innovation_tracker)
        
        # Toggle connection
        if random.random() < self.config.mutate_toggle_connection_rate:
            if self.genes:
                gene = random.choice(self.genes)
                gene.enabled = not gene.enabled
        
        # Clean up any non-feedforward connections that might have been created
        self.cleanup_non_feedforward_connections()
    
    def _add_connection(self, innovation_tracker):
        """Add a new connection between two unconnected nodes"""
        nodes = list(self.get_nodes())
        if len(nodes) < 2:
            return
        
        # Separate nodes by type for proper feedforward connections
        input_nodes = [n for n in nodes if n < self.config.input_size or n == -1]  # Include bias
        hidden_nodes = [n for n in nodes if n >= self.config.input_size + self.config.output_size]
        output_nodes = [n for n in nodes if self.config.input_size <= n < self.config.input_size + self.config.output_size]
        
        # Determine actual layers to respect layer structure
        input_layer = set(input_nodes)
        output_layer = set(output_nodes)
        all_nodes = self.get_nodes()
        hidden_nodes_set = set(n for n in all_nodes if n >= self.config.input_size + self.config.output_size)
        layers = self._determine_network_layers(input_layer, hidden_nodes_set, output_layer)
        
        # Try to find two unconnected nodes with proper feedforward structure
        attempts = 0
        while attempts < 200:  # Increased attempts for better coverage
            
            # Strategy 1: Connect from input layer
            if input_nodes and len(layers) > 1:  # More than just input layer
                from_node = random.choice(input_nodes)
                # Find a valid target layer (not input layer)
                valid_target_layers = [layer for layer in layers[1:] if layer]  # Skip input layer
                if valid_target_layers:
                    target_layer = random.choice(valid_target_layers)
                    to_node = random.choice(list(target_layer))
                else:
                    attempts += 1
                    continue
            
            # Strategy 2: Connect between hidden layers (forward only)
            elif len(layers) > 2:  # At least input, hidden, output layers
                # Choose a hidden layer (not first or last)
                hidden_layer_indices = list(range(1, len(layers) - 1))
                if len(hidden_layer_indices) > 1:  # Multiple hidden layers
                    from_layer_idx = random.choice(hidden_layer_indices[:-1])  # Not the last hidden layer
                    to_layer_idx = from_layer_idx + 1  # Next layer forward
                    
                    from_layer = layers[from_layer_idx]
                    to_layer = layers[to_layer_idx]
                    
                    if from_layer and to_layer:
                        from_node = random.choice(list(from_layer))
                        to_node = random.choice(list(to_layer))
                    else:
                        attempts += 1
                        continue
                else:
                    attempts += 1
                    continue
            
            # Strategy 3: Connect from last hidden layer to output
            elif len(layers) > 2:  # Has hidden layers
                last_hidden_layer = layers[-2]  # Second to last layer (last hidden)
                output_layer = layers[-1]  # Last layer (output)
                
                if last_hidden_layer and output_layer:
                    from_node = random.choice(list(last_hidden_layer))
                    to_node = random.choice(list(output_layer))
                else:
                    attempts += 1
                    continue
            else:
                attempts += 1
                continue
            
            # Check if connection already exists
            connection_exists = any(g.from_node == from_node and g.to_node == to_node 
                                 for g in self.genes)
            
            # Ensure purely feedforward: no connections between nodes at the same layer
            # and no backward connections
            is_feedforward = self._is_feedforward_connection(from_node, to_node)
            
            if not connection_exists and from_node != to_node and is_feedforward:
                innovation = innovation_tracker.get_innovation_number(from_node, to_node)
                weight = random.gauss(0, 1.0)
                self.genes.append(Gene(innovation, from_node, to_node, weight))
                break
            
            attempts += 1
    
    def cleanup_non_feedforward_connections(self):
        """Remove any non-feedforward connections from the genome"""
        if not self.config.cleanup_non_feedforward:
            return
        
        # Get all nodes and their layers
        input_nodes = set(range(self.config.input_size)) | {-1}  # Include bias
        output_nodes = set(range(self.config.input_size, self.config.input_size + self.config.output_size))
        all_nodes = self.get_nodes()
        hidden_nodes = set(n for n in all_nodes if n >= self.config.input_size + self.config.output_size)
        
        # Determine actual layers
        layers = self._determine_network_layers(input_nodes, hidden_nodes, output_nodes)
        
        # Check and disable non-feedforward connections
        connections_to_disable = []
        for gene in self.genes:
            if not gene.enabled:
                continue
                
            from_layer_idx = self._get_node_layer_index(gene.from_node, layers)
            to_layer_idx = self._get_node_layer_index(gene.to_node, layers)
            
            # Disable if same layer or backward connection
            if from_layer_idx >= to_layer_idx:
                connections_to_disable.append(gene)
        
        # Disable the problematic connections
        for gene in connections_to_disable:
            gene.enabled = False
    
    def _is_feedforward_connection(self, from_node: int, to_node: int) -> bool:
        """Check if a connection is feedforward (no cycles, no same-layer connections)"""
        # Define node layers
        input_layer = set(range(self.config.input_size)) | {-1}  # Include bias
        output_layer = set(range(self.config.input_size, self.config.input_size + self.config.output_size))
        
        # Get all hidden nodes
        all_nodes = self.get_nodes()
        hidden_nodes = set(n for n in all_nodes if n >= self.config.input_size + self.config.output_size)
        
        # Determine actual layers using topological analysis
        layers = self._determine_network_layers(input_layer, hidden_nodes, output_layer)
        
        # Check if both nodes are in the same layer
        from_layer_idx = self._get_node_layer_index(from_node, layers)
        to_layer_idx = self._get_node_layer_index(to_node, layers)
        
        if from_layer_idx == to_layer_idx:
            return False  # Same layer connection not allowed
        
        # Check for backward connections
        if from_layer_idx >= to_layer_idx:
            return False  # Backward connection not allowed
        
        return True
    
    def _determine_network_layers(self, input_nodes: set, hidden_nodes: set, output_nodes: set) -> list:
        """Determine actual network layers using topological analysis"""
        # Start with input layer
        layers = [input_nodes]
        
        # Build adjacency list for hidden nodes
        adjacency = {node: set() for node in hidden_nodes}
        for gene in self.genes:
            if gene.enabled and gene.from_node in hidden_nodes and gene.to_node in hidden_nodes:
                adjacency[gene.from_node].add(gene.to_node)
        
        # Find nodes with no incoming connections (first hidden layer)
        current_layer = set()
        for node in hidden_nodes:
            has_incoming = False
            for gene in self.genes:
                if gene.enabled and gene.to_node == node and gene.from_node in hidden_nodes:
                    has_incoming = True
                    break
            if not has_incoming:
                current_layer.add(node)
        
        # Build layers using topological sort
        remaining_nodes = hidden_nodes - current_layer
        while current_layer:
            layers.append(current_layer)
            next_layer = set()
            
            # Find nodes that can be reached from current layer
            for node in remaining_nodes:
                can_reach = False
                for gene in self.genes:
                    if gene.enabled and gene.from_node in current_layer and gene.to_node == node:
                        can_reach = True
                        break
                if can_reach:
                    next_layer.add(node)
            
            current_layer = next_layer
            remaining_nodes -= next_layer
        
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
    
    def _add_node(self, innovation_tracker):
        """Add a new node by splitting an existing connection"""
        if not self.genes:
            return
        
        # Choose a random enabled connection to split
        enabled_genes = [g for g in self.genes if g.enabled]
        if not enabled_genes:
            return
        
        gene_to_split = random.choice(enabled_genes)
        gene_to_split.enabled = False
        
        # Create new node
        new_node_id = max(self.get_nodes()) + 1
        
        # Create two new connections
        innovation1 = innovation_tracker.get_innovation_number(gene_to_split.from_node, new_node_id)
        innovation2 = innovation_tracker.get_innovation_number(new_node_id, gene_to_split.to_node)
        
        self.genes.append(Gene(innovation1, gene_to_split.from_node, new_node_id, 1.0))
        self.genes.append(Gene(innovation2, new_node_id, gene_to_split.to_node, gene_to_split.weight))
    
    def _would_create_cycle(self, from_node: int, to_node: int) -> bool:
        """Check if adding a connection would create a cycle (non-feedforward)"""
        return not self._is_feedforward_connection(from_node, to_node)
    
    def distance(self, other: 'Genome') -> float:
        """Calculate distance between two genomes for speciation"""
        # Count disjoint and excess genes
        disjoint = 0
        excess = 0
        weight_diff = 0.0
        matching_genes = 0
        
        # Sort genes by innovation number
        self_genes = sorted(self.genes, key=lambda g: g.innovation_number)
        other_genes = sorted(other.genes, key=lambda g: g.innovation_number)
        
        i = 0
        j = 0
        
        while i < len(self_genes) and j < len(other_genes):
            if self_genes[i].innovation_number < other_genes[j].innovation_number:
                disjoint += 1
                i += 1
            elif self_genes[i].innovation_number > other_genes[j].innovation_number:
                disjoint += 1
                j += 1
            else:  # Matching innovation numbers
                weight_diff += abs(self_genes[i].weight - other_genes[j].weight)
                matching_genes += 1
                i += 1
                j += 1
        
        # Add remaining genes as excess
        excess += len(self_genes) - i + len(other_genes) - j
        
        # Normalize
        N = max(len(self_genes), len(other_genes))
        if N < 20:
            N = 1
        
        distance = (self.config.excess_coefficient * excess + 
                   self.config.disjoint_coefficient * disjoint) / N
        
        if matching_genes > 0:
            distance += self.config.weight_coefficient * weight_diff / matching_genes
        
        return distance
    
    def crossover(self, other: 'Genome') -> 'Genome':
        """Create a new genome by crossing over with another genome"""
        child = Genome(self.config)
        child.genes = []
        
        # Sort genes by innovation number
        self_genes = sorted(self.genes, key=lambda g: g.innovation_number)
        other_genes = sorted(other.genes, key=lambda g: g.innovation_number)
        
        i = 0
        j = 0
        
        while i < len(self_genes) and j < len(other_genes):
            if self_genes[i].innovation_number < other_genes[j].innovation_number:
                # Disjoint gene from self
                child.genes.append(self_genes[i].copy())
                i += 1
            elif self_genes[i].innovation_number > other_genes[j].innovation_number:
                # Disjoint gene from other
                child.genes.append(other_genes[j].copy())
                j += 1
            else:
                # Matching gene - randomly choose from either parent
                if random.random() < 0.5:
                    child.genes.append(self_genes[i].copy())
                else:
                    child.genes.append(other_genes[j].copy())
                i += 1
                j += 1
        
        # Add remaining genes from self (excess genes)
        while i < len(self_genes):
            child.genes.append(self_genes[i].copy())
            i += 1
        
        # Add remaining genes from other (excess genes)
        while j < len(other_genes):
            child.genes.append(other_genes[j].copy())
            j += 1
        
        # Reset fitness for new child
        child.fitness = 0.0
        child.adjusted_fitness = 0.0
        child.species_id = None
        
        # Clean up any non-feedforward connections in the child
        child.cleanup_non_feedforward_connections()
        
        return child
    
    def copy(self):
        """Create a deep copy of the genome"""
        new_genome = Genome(self.config)
        new_genome.genes = [gene.copy() for gene in self.genes]
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        new_genome.species_id = self.species_id
        return new_genome

class InnovationTracker:
    """Tracks innovation numbers for historical markings"""
    def __init__(self):
        self.innovation_counter = 0
        self.innovation_history = {}  # (from_node, to_node) -> innovation_number
    
    def get_innovation_number(self, from_node: int, to_node: int) -> int:
        """Get or create innovation number for a connection"""
        key = (from_node, to_node)
        if key not in self.innovation_history:
            self.innovation_history[key] = self.innovation_counter
            self.innovation_counter += 1
        return self.innovation_history[key]

class Species:
    """Represents a species of similar genomes"""
    def __init__(self, representative: Genome):
        self.representative = representative
        self.members: List[Genome] = [representative]
        self.fitness_history = deque(maxlen=10)
        self.stagnation = 0
    
    def add_member(self, genome: Genome):
        self.members.append(genome)
    
    def remove_member(self, genome: Genome):
        if genome in self.members:
            self.members.remove(genome)
    
    def update_fitness(self):
        """Update species fitness and check for stagnation"""
        if not self.members:
            return
        
        # Calculate average fitness
        avg_fitness = sum(m.fitness for m in self.members) / len(self.members)
        self.fitness_history.append(avg_fitness)
        
        # Check for stagnation
        if len(self.fitness_history) >= 2:
            if self.fitness_history[-1] <= self.fitness_history[-2]:
                self.stagnation += 1
            else:
                self.stagnation = 0
    
    def get_offspring_count(self, population_size: int, total_adjusted_fitness: float) -> int:
        """Calculate how many offspring this species should produce"""
        if total_adjusted_fitness == 0:
            return 1
        
        species_adjusted_fitness = sum(m.adjusted_fitness for m in self.members)
        return max(1, int(species_adjusted_fitness / total_adjusted_fitness * population_size)) 