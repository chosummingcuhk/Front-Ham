from abc import ABC


class AbstractAgent(ABC):
    def __init__(self, color ):
        self.color = color
    def action(self, state):
        pass

import random
from collections import Counter

class FrontHamEnv:
    def __init__(self):
        self.colors = list(range(5))  # 0 to 4
        self.players = self.colors.copy()  # Each player's ID is their color
        self.reset()

    def reset(self):
        """Reset the game environment to an initial state."""
        self.sock_counts = {c: 6 for c in self.colors}
        self.active_players = set(self.players)
        self.turn_order = random.sample(self.players, k=len(self.players))
        self.action_history = []
        self.acted_players = set()
        self.done = False
        self.current_player = self.get_next_player()
        return self.get_observation(self.current_player)

    def get_observation(self, player):
        """Return observation for the given player."""
        return {
            'sock_counts': self.sock_counts.copy(),
            'own_color': player,
            'action_history': self.action_history.copy()
        }

    def get_next_player(self):
        """Return the next active player in turn order."""
        for p in self.turn_order:
            if p in self.active_players and p not in self.acted_players:
                return p
        return None

    def is_valid_action(self, player, removals, add_color):
        """Check if the action is valid."""
        if player not in self.active_players:
            return False
        if len(removals) != 3:
            return False
        if any(c not in self.colors for c in removals):
            return False
        if add_color not in self.colors:
            return False

        removal_counts = Counter(removals)
        for c, cnt in removal_counts.items():
            if self.sock_counts[c] < cnt:
                return False
        return True

    def apply_action(self, removals, add_color):
        """Apply the action to the game state."""
        removal_counts = Counter(removals)
        for c, cnt in removal_counts.items():
            self.sock_counts[c] -= cnt
        self.sock_counts[add_color] += 1

    def check_eliminations(self, immediate=False):
        """Check and apply eliminations."""
        eliminated = []
        for c in self.colors:
            if self.sock_counts[c] == 0 and c in self.active_players:
                eliminated.append(c)
        for c in eliminated:
            self.active_players.remove(c)

        # Immediate elimination for final round
        if immediate:
            return

    def step(self, player, removals, add_color):
        """
        Process an action from the current player.
        Returns: next_observation, reward, done, info
        """
        if player != self.current_player:
            raise ValueError("Not the current player's turn.")

        # Validate action
        if not self.is_valid_action(player, removals, add_color):
            raise ValueError("Invalid action.")

        # Apply the action
        self.apply_action(removals, add_color)

        # Record the action
        self.action_history.append((player, removals, add_color))
        self.acted_players.add(player)

        # Immediate elimination check (for final round)
        immediate_elimination = False
        if len(self.active_players) <= 2:
            immediate_elimination = True
            self.check_eliminations(immediate=True)

        # End of round check
        current_round_players = [p for p in self.turn_order if p in self.active_players]
        if self.acted_players == set(current_round_players) or self.done:
            # Deferred elimination
            if not immediate_elimination:
                self.check_eliminations()

            # Reset round tracking
            self.acted_players.clear()
            current_round_players = [p for p in self.turn_order if p in self.active_players]

        # Check if game is over
        if len(self.active_players) <= 1:
            self.done = True

        # Determine next player
        self.current_player = self.get_next_player()
        reward = 1 if len(self.active_players) == 1 and self.current_player is None else 0

        # Return next observation
        next_obs = self.get_observation(self.current_player) if self.current_player is not None else {}
        return next_obs, reward, self.done, {}