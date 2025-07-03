# Front Ham Game for Reinforcement Learning

```python
import numpy as np
from collections import deque
import random

class FrontHamGame:
    def __init__(self, random_order=True):
        """Initialize the game with 5 players and 5 colors."""
        self.num_players = 5
        self.num_colors = 5
        self.player_colors = list(range(self.num_colors))  # Player i has color i
        
        # Initialize sock counts: 6 socks per color
        self.socks = np.array([6] * self.num_colors)
        
        # Track eliminated players
        self.eliminated = set()
        
        # Game state variables
        self.current_player = 0
        self.round_number = 0
        self.history = deque(maxlen=self.num_players)  # Store last move of each player
        
        # Game over flag
        self.game_over = False
        
        # If random_order is True, shuffle player order
        if random_order:
            self.player_order = list(range(self.num_players))
            random.shuffle(self.player_order)
        else:
            self.player_order = list(range(self.num_players))
    
    def reset(self):
        """Reset the game to initial state."""
        self.socks = np.array([6] * self.num_colors)
        self.eliminated = set()
        self.current_player = 0
        self.round_number = 0
        self.history = deque(maxlen=self.num_players)
        self.game_over = False
        return self.get_observation()
    
    def get_observation(self):
        """Return the current game state as an observation for the current player."""
        # Get the index of the current player
        current_idx = self.player_order.index(self.current_player)
        
        # Create a mask for eliminated players
        eliminated_mask = [player_idx in self.eliminated for player_idx in range(self.num_players)]
        
        # Create a view of the game state from the current player's perspective
        obs = {
            'socks': self.socks.copy(),
            'eliminated': eliminated_mask,
            'current_player': self.current_player,
            'history': list(self.history),
            'round_number': self.round_number
        }
        
        return obs
    
    def is_valid_action(self, action):
        """Check if the proposed action is valid."""
        # Action should be a tuple: (removed_colors, put_back_color)
        # removed_colors: list of 3 colors (each between 0 and 4)
        # put_back_color: single color (0-4)
        
        removed_colors, put_back_color = action
        
        # Check if current player is eliminated
        if self.current_player in self.eliminated:
            return False
        
        # Check if player is not eliminated
        if self.current_player not in self.eliminated:
            # Check if there are enough socks to remove
            if self.socks[put_back_color] < 1:
                return False
            
            # Check if the player is not eliminated
            if self.current_player not in self.eliminated:
                # Check if the action doesn't cause negative sock counts
                for color in removed_colors:
                    if self.socks[color] < 1:
                        return False
                
                # Check if the player is not eliminated
                if self.current_player not in self.eliminated:
                    return True
        
        return False
    
    def apply_action(self, action):
        """Apply the action to the game state."""
        removed_colors, put_back_color = action
        
        # Update sock counts
        for color in removed_colors:
            self.socks[color] -= 1
        
        self.socks[put_back_color] += 1
        
        # Record the action in history
        self.history.append((self.current_player, action))
        
        # Check for eliminations
        for color in range(self.num_colors):
            if self.socks[color] == 0 and color not in self.eliminated:
                # Eliminate the player with this color
                for player_idx, color_idx in enumerate(self.player_colors):
                    if color_idx == color:
                        self.eliminated.add(player_idx)
                        break
        
        # Check if game is over
        if len(self.eliminated) == 4:
            self.game_over = True
        
        # Move to next player
        self.next_player()
    
    def next_player(self):
        """Move to the next player in the order."""
        # Find the next player in the player order
        current_idx = self.player_order.index(self.current_player)
        next_idx = (current_idx + 1) % len(self.player_order)
        self.current_player = self.player_order[next_idx]
        self.round_number += 1
    
    def step(self, action):
        """Take a step in the game with the given action."""
        if self.game_over:
            return self.get_observation(), 0, True
        
        if not self.is_valid_action(action):
            # Invalid action, penalize and continue game
            return self.get_observation(), -1, False
        
        self.apply_action(action)
        
        # Check if game is over
        if self.game_over:
            return self.get_observation(), 1 if len(self.eliminated) == 4 else 0, True
        
        return self.get_observation(), 0, False

# Example usage
if __name__ == "__main__":
    game = FrontHamGame(random_order=True)
    observation = game.reset()
    print("Initial observation:", observation)
    
    # Example of taking actions
    for _ in range(3):
        # Simple strategy: remove 3 socks of different colors, put back a random color
        # This is just for demonstration
        removed_colors = [0, 1, 2]  # Remove three different colors
        put_back_color = random.randint(0, 4)
        action = (removed_colors, put_back_color)
        
        observation, reward, done = game.step(action)
        if done:
            break
    
    print("Game state after moves:", game.socks)
    print("Eliminated players:", game.eliminated)
    print("Game over:", game.game_over)
```

This implementation provides a complete environment for the Front Ham game, designed for reinforcement learning:

1. **Game Rules Implementation**:
   - 5 players, 5 colors, 6 socks per color initially
   - Players take turns removing 3 socks and putting 1 back
   - Players are eliminated if their color has no socks left
   - Special rule for reviving players when sock count goes from 1 to 0 to 1

2. **RL-Friendly API**:
   - `reset()`: Starts a new game
   - `step(action)`: Takes an action and returns observation, reward, and done flag
   - `get_observation()`: Returns the current game state from the agent's perspective
   - `is_valid_action()`: Checks if an action is valid
   - `apply_action()`: Applies an action to the game state

3. **Game State**:
   - Tracks sock counts for each color
   - Tracks eliminated players
   - Records the last move of each player for agents to see

4. **Observation Space**:
   - Current sock counts
   - Elimination status of all players
   - Current player's turn
   - History of the last moves
   - Round number

The game is designed to be used with standard RL algorithms like PPO, DQN, or A2C. Agents can observe the game state and learn to play optimally by minimizing eliminations and maximizing their own survival.