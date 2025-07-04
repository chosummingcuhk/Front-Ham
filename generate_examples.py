import random
import io
import statistics
game_length = []
#main loop for multiple batches of text
try:
    for _ in range(10000):
        result = io.StringIO()
        #generate a number of games
        for _ in range(10000):
            game_length.append(0)
            result.write('r6o6y6g6b6|')
            #logic for one game
            state = [['red', 6], ['orange', 6], ['yellow', 6], ['green', 6], ['blue', 6]]
            while len(state) > 2:
                i = random.randrange(len(state))
                state[i][1] += 1
                for _ in range(3):
                    i = random.randrange(len(state))
                    if state[i][1] > 1:
                        state[i][1] -= 1
                    else:
                        del state[i]
                for x in state:
                    result.write(x[0][0])
                    result.write(f'{x[1]:x}')
                result.write('|')
                game_length[-1] += 1
            #endgame
            if len(state) == 2:
                result.write('!')
                if state[0][1] > state[1][1]:
                    result.write(state[0][0][0])
                elif state[1][1] > state[0][1]:
                    result.write(state[1][0][0])
                else:
                    result.write('d')
            if len(state) == 1:
                result.write('!')
                result.write(state[0][0][0])
            #write to string
            result.write('\n')
        with open('example data', 'a') as f:
            #write one batch
            f.write(result.getvalue())
except KeyboardInterrupt:
    print(min(game_length[:-2]))
    print(max(game_length))
    print(statistics.mean(game_length))
    print(statistics.stdev(game_length))