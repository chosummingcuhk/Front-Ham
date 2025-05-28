import numpy as np
import random
from collections import defaultdict
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


class FrontHamGame(Env):
    COLORS = ['red', 'blue', 'green', 'yellow', 'purple']
    NO_OP = 'no_op'

    def __init__(self):
        # Action space: 3 removals + 1 addition (each action: 5 colors + 1 no-op)
        self.action_space = Box(low=0, high=1, shape=(5,))  # Simplified for demonstration

        # Observation space: sock counts (0-6 per color) + own color index
        self.observation_space = Box(low=0, high=6, shape=(5,))

        self.reset()

    def reset(self):
        """Initialize game state"""
        # Game state
        self.sock_counts = {color: 6 for color in self.COLORS}
        self.players = {i: {'color': None, 'active': True} for i in range(5)}
        self._assign_colors()
        self.turn_order = list(range(5))
        random.shuffle(self.turn_order)
        self.current_turn = 0
        self.final_phase = False
        return self._get_observation()

    def _assign_colors(self):
        """Assign unique colors to players"""
        available_colors = self.COLORS.copy()
        for pid in self.players:
            color = random.choice(available_colors)
            self.players[pid]['color'] = color
            available_colors.remove(color)

    def _get_observation(self):
        """Return observation for current player"""
        sock_array = np.array([self.sock_counts[color] for color in self.COLORS])
        return sock_array

    def step(self, action):
        """
        Execute action and return (observation, reward, done, info)

        Args:
            action: dict with 'removals' and 'addition' keys

        Returns:
            tuple: (observation, reward, done, info)
        """
        current_player = self.turn_order[self.current_turn]
        own_color = self.players[current_player]['color']

        # Validate and execute action
        if self.final_phase:
            valid, action = self._validate_final_action(action, own_color)
        else:
            valid, action = self._validate_standard_action(action, own_color)

        if not valid:
            return self._get_observation(), -1, True, {'invalid_action': True}

        # Apply action
        for removal in action['removals']:
            if removal != self.NO_OP:
                self.sock_counts[removal] -= 1

        if action['addition'] != self.NO_OP:
            self.sock_counts[action['addition']] += 1

        # Check eliminations
        eliminated = self._check_eliminations()

        # Check if game ended
        active_players = [p for p in self.players if self.players[p]['active']]
        done = len(active_players) == 1

        # Calculate reward
        reward = 0
        if done:
            if self.players[current_player]['active']:
                reward = 1  # Win
            else:
                reward = -1  # Loss

        # Next turn
        self.current_turn = (self.current_turn + 1) % len(self.turn_order)

        # Update final phase status
        if len(active_players) <= 2 and not self.final_phase:
            self.final_phase = True

        return self._get_observation(), reward, done, {'eliminated': eliminated}

    def _validate_standard_action(self, action, own_color):
        """Validate action for standard play phase"""
        # Ensure 3 removals and 1 addition
        if len(action['removals']) != 3:
            return False, None

        # Check for invalid removals
        removal_counts = defaultdict(int)
        for color in action['removals']:
            if color != self.NO_OP:
                removal_counts[color] += 1
                if self.sock_counts[color] < removal_counts[color]:
                    return False, None

        # Check addition validity
        if action['addition'] != self.NO_OP:
            if self.sock_counts[action['addition']] == 0:
                return False, None

        return True, action

    def _validate_final_action(self, action, own_color):
        """Validate action for final phase"""
        # No-Ops are allowed, but must result in elimination
        removals = [r for r in action['removals'] if r != self.NO_OP]

        # Check if removals target single color
        if len(set(removals)) > 1:
            return False, None

        if removals:
            target_color = removals[0]
            required_removals = self.sock_counts[target_color]

            # Check if action matches valid No-Op patterns
            if required_removals == 1 and len(removals) == 1:
                return True, {'removals': [target_color, self.NO_OP, self.NO_OP],
                              'addition': self.NO_OP}

            if required_removals == 2 and len(removals) == 2:
                return True, {'removals': [target_color, target_color, self.NO_OP],
                              'addition': self.NO_OP}

            if required_removals == 3 and len(removals) == 3:
                return True, {'removals': [target_color, target_color, target_color],
                              'addition': self.NO_OP}

        # Standard 3+1 actions still allowed
        return self._validate_standard_action(action, own_color)

    def _check_eliminations(self):
        """Check and process eliminations"""
        eliminated = []
        for color in self.sock_counts:
            if self.sock_counts[color] == 0:
                for pid in self.players:
                    if self.players[pid]['color'] == color:
                        self.players[pid]['active'] = False
                        eliminated.append(pid)
        return eliminated

    def render(self, mode='human'):
        """Display game state"""
        print(f"Sock counts: {self.sock_counts}")
        print(f"Active players: {[p for p in self.players if self.players[p]['active']]}")


# Example usage
if __name__ == "__main__":
    env = FrontHamGame()
    obs = env.reset()

    for _ in range(100):  # Limit episodes for testing
        action = {
            'removals': random.sample(env.COLORS, 3),
            'addition': random.choice(env.COLORS)
        }
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            print("Game ended!")
            break