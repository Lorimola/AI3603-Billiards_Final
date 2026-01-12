# Imitation Learning Module
from .model import BilliardsPolicyNetwork, BilliardsPolicyNetworkSmall, create_model
from .agent_imitation import ImitationAgent, NewAgent

__all__ = [
    'BilliardsPolicyNetwork',
    'BilliardsPolicyNetworkSmall', 
    'create_model',
    'ImitationAgent',
    'NewAgent',
]











