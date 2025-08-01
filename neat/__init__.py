"""
NEAT (NeuroEvolution of Augmenting Topologies) Implementation for Slime Volleyball

This package contains a complete implementation of the NEAT algorithm from first principles.
"""

from .neat_core import NEATConfig, Gene, Genome, InnovationTracker, Species
from .neat_network import NEATNetwork
from .neat_population import NEATPopulation
from .neat_visualization import NEATVisualizer

__version__ = "1.0.0"
__author__ = "NEAT Implementation Team"

__all__ = [
    'NEATConfig',
    'Gene', 
    'Genome',
    'InnovationTracker',
    'Species',
    'NEATNetwork',
    'NEATPopulation',
    'NEATVisualizer'
] 