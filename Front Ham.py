from enum import Enum, auto
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass
from copy import deepcopy


class Color(Enum):
    Red = auto()
    Blue = auto()
    Green = auto()
    Yellow = auto()
    Purple = auto()


@dataclass(frozen=True)
class GameState:
    round_number: int
    sock_counts: Dict[Color, int]
    active_players: List[Color]
    pending_eliminations: List[Color]
    current_player: Color


@dataclass(frozen=True)
class GameAction:
    removals: Tuple[Color, Color, Color]
    addition: Color


class Front_Ham:
    def __init__(self):
        self.colors = list(Color)
        self.players = {color: 6 for color in self.colors}
        self.active_players = self.colors.copy()
        self.pending_eliminations = set()
        self.round_number = 0
        self.current_player_idx = 0

    def reset(self) -> GameState:
        self.__init__()
        return self.current_state

    @property
    def current_state(self) -> GameState:
        return GameState(
            round_number=self.round_number,
            sock_counts=deepcopy(self.players),
            active_players=deepcopy(self.active_players),
            pending_eliminations=deepcopy(list(self.pending_eliminations)),
            current_player=self.current_player
        )

    @property
    def current_player(self) -> Color:
        return self.active_players[self.current_player_idx]

    def get_valid_actions(self) -> List[GameAction]:
        valid_actions = []

        # Generate valid removal sequences with proper count tracking
        def generate_removals(current_counts, depth, current_sequence):
            if depth == 0:
                return [current_sequence]
            combinations = []
            for color in self.colors:
                if current_counts[color] > 0:
                    new_counts = deepcopy(current_counts)
                    new_counts[color] -= 1
                    combinations += generate_removals(
                        new_counts,
                        depth - 1,
                        current_sequence + (color,)
                    )
            return combinations

        # Start with current sock counts
        initial_counts = deepcopy(self.players)
        removal_sequences = generate_removals(initial_counts, 3, ())

        # Generate full actions with valid additions
        for removals in removal_sequences:
            for add_color in self.colors:
                valid_actions.append(GameAction(
                    removals=removals,
                    addition=add_color
                ))

        return valid_actions

    def step(self, action: GameAction) -> GameState:
        # Create temporary copy for validation
        temp_counts = deepcopy(self.players)

        # Validate removals in sequence
        for color in action.removals:
            if temp_counts[color] < 1:
                raise ValueError(f"Invalid removal: {color} has {self.players[color]} socks")
            temp_counts[color] -= 1

        # Apply valid action to actual game state
        for color in action.removals:
            self.players[color] -= 1
            if self.players[color] == 0:
                self.pending_eliminations.add(color)

        self.players[action.addition] += 1

        # Move to next player
        self.current_player_idx = (self.current_player_idx + 1) % len(self.active_players)

        # Process eliminations at round end
        if self.current_player_idx == 0:
            self._process_eliminations()
            self.round_number += 1

        return self.current_state

    def _process_eliminations(self):
        new_active = []
        for color in self.active_players:
            if self.players[color] > 0:
                new_active.append(color)

        # Check for revivals
        revived = []
        for color in self.pending_eliminations:
            if color not in new_active and self.players[color] > 0:
                new_active.append(color)
                revived.append(color)

        self.active_players = new_active
        self.pending_eliminations = set()


# Example usage
if __name__ == "__main__":
    game = Front_Ham()
    state = game.reset()

    try:
        while len(game.active_players) > 1:
            print(f"\nCurrent Player: {game.current_player.name}")
            valid_actions = game.get_valid_actions()
            action = random.choice(valid_actions)
            new_state = game.step(action)

            print(f"Action: Removed {[r.name for r in action.removals]}, Added {action.addition.name}")
            print("Sock Counts:")
            for color, count in game.players.items():
                print(f"{color.name}: {count}")

        print(f"\nWinner: {game.active_players[0].name}")
    except ValueError as e:
        print(f"Error: {e}")
        print("This should never happen with properly generated valid actions!")