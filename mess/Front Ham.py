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