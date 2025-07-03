import random

class FrontHamGame:
    def __init__(self, players):
        self.players = players
        self.colors = ['Red', 'Blue', 'Green', 'Yellow', 'Purple']
        self.sock_counts = {color: 6 for color in self.colors}
        self.player_colors = {player: color for player, color in zip(players, self.colors)}
        self.turn_order = random.sample(players, len(players))
        self.eliminated = set()
        self.current_turn = 0

    def get_valid_actions(self, player):
        if len(self.eliminated) >= 3:
            return self.get_final_phase_actions(player)
        else:
            return self.get_standard_phase_actions(player)

    def get_standard_phase_actions(self, player):
        actions = []
        for color1 in self.colors:
            for color2 in self.colors:
                for color3 in self.colors:
                    for add_color in self.colors:
                        if color1 != color2 or color2 != color3 or color3 != color1:
                            removals = [color1, color2, color3]
                            if self.is_valid_removals(removals) and self.is_valid_addition(add_color, removals):
                                actions.append((removals, add_color))
        return actions

    def get_final_phase_actions(self, player):
        actions = []
        for color in self.colors:
            for no_op_count in range(3):
                removals = [color] * (3 - no_op_count) + ['No-Op'] * no_op_count
                if self.is_valid_removals(removals) and self.is_valid_addition('No-Op', removals):
                    actions.append((removals, 'No-Op'))
        return actions

    def is_valid_removals(self, removals):
        for color in removals:
            if color != 'No-Op' and self.sock_counts[color] <= 0:
                return False
        return True

    def is_valid_addition(self, add_color, removals):
        if add_color != 'No-Op' and self.sock_counts[add_color] <= 0:
            return False
        return True

    def apply_action(self, player, action):
        removals, addition = action
        for color in removals:
            if color != 'No-Op':
                self.sock_counts[color] -= 1
        if addition != 'No-Op':
            self.sock_counts[addition] += 1

        self.check_eliminations()

    def check_eliminations(self):
        for color in self.colors:
            if self.sock_counts[color] <= 0 and color not in self.eliminated:
                self.eliminated.add(color)
                for player, p_color in self.player_colors.items():
                    if p_color == color:
                        self.players.remove(player)
                        break

    def next_turn(self):
        self.current_turn = (self.current_turn + 1) % len(self.turn_order)

    def play_turn(self, player, action):
        if player not in self.players:
            raise ValueError("Player is eliminated or not in the game.")
        if action not in self.get_valid_actions(player):
            raise ValueError("Invalid action for the current game state.")

        self.apply_action(player, action)
        self.next_turn()

    def is_game_over(self):
        return len(self.players) <= 1

    def get_winner(self):
        if self.is_game_over():
            return self.players[0] if self.players else None
        return None

# Example usage
players = ['Player1', 'Player2', 'Player3', 'Player4', 'Player5']
game = FrontHamGame(players)

# Simulate a turn
valid_actions = game.get_valid_actions(players[0])
action = valid_actions[0]  # Choose the first valid action
game.play_turn(players[0], action)

# Check if the game is over
if game.is_game_over():
    winner = game.get_winner()
    print(f"The winner is {winner}")
else:
    print("The game is still ongoing.")