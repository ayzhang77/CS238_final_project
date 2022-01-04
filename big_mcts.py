"""
Author: Amy Zhang
Class: CS238 Decision Making Under Uncertainty
Date: 10/31/2020

Description: Simulates simplified version of Plants versus Zombies in a big 10x10 world that uses Monte Carlo Tree Search
(MCTS) to generate action at each state in the game.

"""

import pandas as pd
import numpy as np
import copy
import math

gamma = 0.95


def containsZombie(state):
    return True if 10 in state or 11 in state or 20 in state else False


def loosingState(state):
    home = [state[0], state[10], state[20], state[30], state[40], state[50], state[60], state[70], state[80],  state[90]]
    return containsZombie(home)


def moveZombies(state):
    eaten = 0
    if not containsZombie(state):
        return eaten
    else:
        indices = [i for i, j in enumerate(state) if j == 10 or j == 11 or j == 20]
        for index in indices:
            if (state[index - 1] == 3):
                # zombie ran into plant and ate it
                eaten += 1
            state[index - 1] = state[index]
            state[index] = 0

        return eaten


def addZombie(state):
    eaten = 0
    landing = [state[9], state[19], state[29], state[39], state[49], state[59], state[69], state[79], state[89], state[99]]
    free = [j for j, k in enumerate(landing) if k == 0 or k == 3]
    if len(free) == 0:
        return eaten
    else:
        index = np.random.choice(free)
        if state[9 + (10*index)] == 3:
            # zombie ran into plant and ate it
            eaten += 1
        # add strong zombie with prob 0.5 and weak zombie with prob 0.5
        state[9 + (10*index)] = np.random.choice([10, 20], p=[0.5, 0.5])

    return eaten


def killZombies(state):
    killed = 0
    if not containsZombie(state):
        return killed

    for i in range(0, 91, 10):
        row = state[i:i+10]
        if containsZombie(row) and 3 in row:
            zombie_indices = [j for j, k in enumerate(row) if k == 10 or k == 11 or k == 20]
            plant_indices = [m for m, n in enumerate(row) if n == 3]
            # if plant is facing zombie, then valid shot
            if zombie_indices[0] > plant_indices[0]:
                zombie = row[zombie_indices[0]]
                # zombie is killed!
                if zombie == 11 or zombie == 20:
                    state[i + zombie_indices[0]] = 0
                    killed += 1
                # strong zombie is weakened
                elif zombie == 10: # strong zombie is weakened
                    state[i + zombie_indices[0]] = 11
    return killed


def simulateGame():
    game = pd.DataFrame(columns = ['s', 'a', 'r', 'sp'])
    win = 0
    state = [0 for i in range(100)]
    # starting state always has one zombie
    addZombie(state)
    step = 0

    #generates (state, action, reward, sp)
    while True:
        state_curr = copy.copy(state)
        tup = [state_curr]

        # ACTION WITH MCTS
        actions = [j for j, k in enumerate(state) if k == 0]
        if len(actions) == 0:
            action = 100
        else:
            # plant with probability 0.5
            action = monteCarloTreeSearch(state, actions, gamma, step) if np.random.random() <= 0.5 else 100

        if action != 100:
            state[action] = 3
        tup.append(action)

        # NEXT STATE
        killed = killZombies(state)
        eaten = 0
        # zombies are clumsy, they successfully step forward with prob 0.5
        if np.random.random() < 0.5:
            eaten += moveZombies(state)
        # next state has new zombie with prob 0.7, zombies can only enter during night
        if np.random.random() < 0.7 and step < 20:
            eaten += addZombie(state)

        # REWARD
        reward = killed - eaten
        # yikes the zombies win
        if loosingState(state):
            reward += -200
            tup.append(reward)
            tup.append(copy.copy(state))
            game.loc[step] = tup
            break
        # wooo! survived the night and killed remaining zombies, humans win!
        if step >= 20 and not containsZombie(state):
            reward += 100
            tup.append(reward)
            tup.append(copy.copy(state))
            game.loc[step] = tup
            win = 1
            break
        tup.append(reward)
        tup.append(copy.copy(state))
        game.loc[step] = tup
        step += 1
    return game, win


def rollout(state, d, gamma, step):
    if d <= 0:
        return 0
    if loosingState(state):
        return -200
    # wooo! survived the night and killed remaining zombies, humans win!
    if step >= 20 and not containsZombie(state):
        return 100

    free = [j for j, k in enumerate(state) if k == 0]

    if len(free) != 0:
        index = np.random.choice(free)
        if step == 0:
            state[index] = 3
        else:
            state[index] = np.random.choice([3, 0], p=[0.5, 0.5])

    reward = getNextState(state, step)

    return reward + gamma * rollout(state, d - 1, gamma, step + 1)


def getNextState(state, step):
    killed = killZombies(state)
    eaten = 0
    # we are smart so we project the worse case of the zombies always moving forward
    eaten += moveZombies(state)
    # next state has new zombie with prob 0.7, zombies can only enter during night
    if np.random.random() < 0.7 and step < 20:
        eaten += addZombie(state)

    # REWARD
    reward = killed - eaten

    return reward


def bonus(Nsa, N_sum):
    return float('inf') if Nsa == 0 else math.sqrt(math.log(N_sum)/Nsa)


def explore(state, actions, N, Q):
    N_sum = sum([N[(tuple(state), a)] for a in actions])

    # HYPERPARAM
    c = 10

    UCB = []
    for action in actions:
        UCB.append(Q[tuple(state), action] + c*bonus(N[tuple(state), action], N_sum))

    index = np.argmax(UCB)
    return actions[index]


def MCTSsimulate(state, depth, Q, N, actions, gamma, step):
    if depth <= 0 or len(actions) == 0:
        return 0

    state_curr = copy.copy(state)
    key = (tuple(state), actions[0])
    if key not in N.keys():
        for action in actions:
            key_to_add = (tuple(state), action)
            N[key_to_add] = 0
            Q[key_to_add] = 0
        x = rollout(state, depth, gamma, step)
        return x

    a = explore(state, actions, N, Q)
    state[a] = 3

    reward = getNextState(state, step) # UPDATES STATE IN PLACE
    sp_actions = [j for j, k in enumerate(state) if k == 0]

    q = reward + gamma*MCTSsimulate(state, depth-1, Q, N, sp_actions, gamma, step)
    N[(tuple(state_curr), a)] += 1
    Q[(tuple(state_curr), a)] += (q - Q[(tuple(state_curr), a)]) / N[(tuple(state_curr), a)]
    return q


def monteCarloTreeSearch(state, actions, gamma, step):
    # REMEMBER TO ADJUST HYPERPARAMS
    k_max = 150
    depth = 20
    N = {}
    Q = {}

    for i in range(k_max):
        state_curr = copy.copy(state)
        MCTSsimulate(state_curr, depth, Q, N, actions, gamma, step)

    best_action = float('-inf')
    best_q = float('-inf')
    for action in actions:
        if (tuple(state), action) in Q:
            if Q[(tuple(state), action)] > best_q:
                best_q = Q[(tuple(state), action)]
                best_action = action

    return best_action


def generateData():
    data = pd.DataFrame(columns=['s', 'a', 'r', 'sp'])
    win_count = 0
    for i in range(1):
        game_tup = simulateGame()
        data = data.append(game_tup[0])
        win_count += game_tup[1]
        print(i, win_count)
    print(win_count)
    return data


def main():
    data = generateData()
    data.to_csv("game_data_big_mcts_single3.csv", index=False)


if __name__ == "__main__":
    main()

