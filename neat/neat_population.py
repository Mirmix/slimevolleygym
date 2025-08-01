"""
NEAT Population Management and Evolution

This module handles the NEAT population, species management, and evolution process.
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from .neat_core import Genome, NEATConfig, InnovationTracker, Species

class NEATPopulation:
    """Manages the NEAT population and evolution"""
    def __init__(self, config: NEATConfig):
        self.config = config
        self.innovation_tracker = InnovationTracker()
        self.species: List[Species] = []
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_genome = None
        
        # Initialize population
        self.population = [Genome(config) for _ in range(config.population_size)]
        self._speciate()
    
    def _speciate(self):
        """Organize genomes into species"""
        # Clear existing species
        for species in self.species:
            species.members.clear()
        self.species = []
        
        # Assign genomes to species
        for genome in self.population:
            assigned = False
            
            for species in self.species:
                if genome.distance(species.representative) < self.config.species_threshold:
                    species.add_member(genome)
                    genome.species_id = id(species)  # Assign species ID
                    assigned = True
                    break
            
            if not assigned:
                # Create new species
                new_species = Species(genome)
                genome.species_id = id(new_species)  # Assign species ID
                self.species.append(new_species)
    
    def _calculate_adjusted_fitness(self):
        """Calculate adjusted fitness for all genomes"""
        for species in self.species:
            for genome in species.members:
                genome.adjusted_fitness = genome.fitness / len(species.members)
    
    def _select_parent(self) -> Genome:
        """Select a parent using tournament selection"""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda g: g.fitness)
    
    def _create_offspring(self) -> Genome:
        """Create a new offspring genome"""
        if random.random() < self.config.crossover_rate and len(self.population) > 1:
            # Crossover
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            child = parent1.crossover(parent2)
        else:
            # Clone
            parent = self._select_parent()
            child = parent.copy()  # Use the copy method
        
        # Mutate
        child.mutate(self.innovation_tracker)
        return child
    
    def evolve(self):
        """Evolve the population for one generation"""
        # Calculate adjusted fitness
        self._calculate_adjusted_fitness()
        
        # Update species fitness and check stagnation
        total_adjusted_fitness = sum(g.adjusted_fitness for g in self.population)
        
        new_population = []
        
        # Elitism: keep best members from each species
        for species in self.species:
            if len(species.members) > 0:
                species.members.sort(key=lambda g: g.fitness, reverse=True)
                elite_count = min(self.config.species_elitism, len(species.members))
                new_population.extend(species.members[:elite_count])
        
        # Create offspring
        while len(new_population) < self.config.population_size:
            offspring = self._create_offspring()
            new_population.append(offspring)
        
        # Update population
        self.population = new_population[:self.config.population_size]
        
        # Update species
        self._speciate()
        
        # Update species fitness
        for species in self.species:
            species.update_fitness()
        
        self.generation += 1
        
        # Update best genome
        for genome in self.population:
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome
    
    def get_statistics(self) -> Dict:
        """Get current population statistics"""
        if not self.population:
            return {}
        
        fitnesses = [g.fitness for g in self.population]
        complexities = [len(g.genes) for g in self.population]
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'species_count': len(self.species),
            'best_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'best_complexity': max(complexities),
            'average_complexity': np.mean(complexities),
            'total_innovations': self.innovation_tracker.innovation_counter
        }
    
    def save_population(self, filename: str):
        """Save the entire population to file"""
        population_data = {
            'config': {
                'population_size': self.config.population_size,
                'species_threshold': self.config.species_threshold,
                'input_size': self.config.input_size,
                'output_size': self.config.output_size
            },
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'innovation_counter': self.innovation_tracker.innovation_counter,
            'innovation_history': self.innovation_tracker.innovation_history,
            'population': []
        }
        
        for genome in self.population:
            genome_data = {
                'fitness': genome.fitness,
                'adjusted_fitness': genome.adjusted_fitness,
                'species_id': genome.species_id,
                'genes': [(g.innovation_number, g.from_node, g.to_node, 
                          g.weight, g.enabled) for g in genome.genes]
            }
            population_data['population'].append(genome_data)
        
        import json
        with open(filename, 'w') as f:
            json.dump(population_data, f, indent=2)
    
    @classmethod
    def load_population(cls, filename: str, config: NEATConfig):
        """Load population from file"""
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        
        population = cls(config)
        population.generation = data['generation']
        population.best_fitness = data['best_fitness']
        population.innovation_tracker.innovation_counter = data['innovation_counter']
        population.innovation_tracker.innovation_history = data['innovation_history']
        
        # Recreate genomes
        population.population = []
        for genome_data in data['population']:
            genome = Genome(config)
            genome.fitness = genome_data['fitness']
            genome.adjusted_fitness = genome_data['adjusted_fitness']
            genome.species_id = genome_data['species_id']
            
            genome.genes = []
            for gene_data in genome_data['genes']:
                innovation, from_node, to_node, weight, enabled = gene_data
                from .neat_core import Gene
                gene = Gene(innovation, from_node, to_node, weight, enabled)
                genome.genes.append(gene)
            
            population.population.append(genome)
        
        population._speciate()
        return population 