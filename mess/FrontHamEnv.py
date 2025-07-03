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
        if player not in self.active_players:
            return False
        if add_color not in self.colors:
            return False

        if len(self.active_players) == 2:
            # Final phase: all removals must be opponent's color
            opponent = next(p for p in self.active_players if p != player)
            if any(c != opponent for c in removals):
                return False
            return True
        else:
            # Standard validation: 3 removals, valid colors
            if len(removals) != 3:
                return False
            if any(c not in self.colors for c in removals):
                return False
            return True

    def apply_action(self, removals, add_color):
        if len(self.active_players) == 2:
            opponent = next(p for p in self.active_players if p != self.current_player)
            # Process removals one-by-one, stopping if opponent is eliminated
            for c in removals:
                if self.sock_counts[c] <= 0:
                    break
                self.sock_counts[c] -= 1
                if self.sock_counts[opponent] == 0:
                    break  # Elimination, stop processing
            # Add only if opponent not eliminated
            if self.sock_counts[opponent] > 0:
                self.sock_counts[add_color] += 1
        else:
            # Standard 3 removals, 1 addition
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

        # Elimination check
        if len(self.active_players) > 2:
            self.check_eliminations()

        # Check if game is over
        if len(self.active_players) <= 1:
            self.done = True

        # Record the action
        self.action_history.append((player, removals, add_color))
        self.acted_players.add(player)


        # Determine next player
        self.current_player = self.get_next_player()
        reward = 1 if len(self.active_players) == 1 and self.current_player is None else 0

        # Return next observation
        next_obs = self.get_observation(self.current_player) if self.current_player is not None else {}
        return next_obs, reward, self.done, {}