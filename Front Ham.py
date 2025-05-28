from copy import deepcopy

class FrontHamGame:
    def __init__(self):
        self.num_players = 5
        self.colors = 5  # 0 to 4
        self.starting_socks = 6
        self.sock_counts = []
        self.active_players = []
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.sock_counts = [self.starting_socks for _ in range(self.colors)]
        self.active_players = [True for _ in range(self.num_players)]
        return self._get_state()

    def _get_state(self):
        """Return current state (sock counts and active players)."""
        return {
            'sock_counts': deepcopy(self.sock_counts),
            'active_players': deepcopy(self.active_players)
        }

    def _is_final_phase(self):
        """Check if current active players are two."""
        return sum(self.active_players) == 2

    def _apply_action(self, action_take, put_color, current_counts):
        """
        Apply an action to current sock counts.

        Args:
            action_take: List of 5 integers >=0 summing to 3.
            put_color: Integer between 0-4.
            current_counts: Current sock counts.

        Returns:
            New sock counts or None if invalid.
        """
        if sum(action_take) != 3:
            return None
        for color in range(self.colors):
            if action_take[color] > current_counts[color]:
                return None
        new_counts = [current_counts[c] - action_take[c] for c in range(self.colors)]
        new_counts[put_color] += 1
        return new_counts

    def _check_eliminations(self):
        """
        Update active_players based on current sock counts.
        Eliminate any active player whose color has 0 socks.
        """
        new_active = self.active_players.copy()
        for color in range(self.num_players):
            if self.active_players[color] and self.sock_counts[color] == 0:
                new_active[color] = False
        self.active_players = new_active

    def _get_active_player_indices(self):
        """Return list of indices of currently active players."""
        return [i for i in range(self.num_players) if self.active_players[i]]

    def step(self, player_index, action_take, put_color):
        """
        Process a player's action.

        Args:
            player_index: Index of the acting player (0-4).
            action_take: List of 5 integers summing to 3.
            put_color: Color to add (0-4).

        Returns:
            state: New state.
            done: Boolean indicating if game ended.
            eliminated: List of eliminated players in this step.
        """
        eliminated = []
        if not self.active_players[player_index]:
            # Player already eliminated, invalid move
            return self._get_state(), False, eliminated

        # Apply the action
        new_counts = self._apply_action(action_take, put_color, self.sock_counts)
        if new_counts is None:
            # Invalid action
            return self._get_state(), False, eliminated

        self.sock_counts = new_counts

        # Check eliminations
        was_final_phase = self._is_final_phase()
        self._check_eliminations()
        current_actives = sum(self.active_players)

        # Detect eliminated players
        for color in range(self.num_players):
            if self.active_players[color] == False and self.sock_counts[color] == 0:
                eliminated.append(color)

        # Check if game is done
        done = current_actives <= 1

        return self._get_state(), done, eliminated

    def is_game_over(self):
        return sum(self.active_players) <= 1

if __name__ == "__main__":
    game = FrontHamGame()
    print("Initial state:", game.sock_counts)

    # Player 0 takes 3 socks from color 0 and puts one back
    action_take = [3, 0, 0, 0, 0]
    put_color = 0
    state, done, eliminated = game.step(0, action_take, put_color)
    print("After action:", game.sock_counts)
    print("Eliminated players:", eliminated)
    print("Game over?", done)

from abstract_agent import FrontHamEnv
env = FrontHamEnv()
obs = env.reset()

while not env.done:
    player = env.current_player
    print(f"Player {player}'s turn. Observing...")
    print(obs)

    # Dummy action: remove 3 socks of color 0, add 1 of color 1
    action = ([0, 0, 0], 1)
    obs, reward, done, info = env.step(player, *action)

print("Game over!")
if reward == 1:
    print(f"Player {next(iter(env.active_players))} won!")