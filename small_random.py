"""
Author: Amy Zhang
Class: CS238 Decision Making Under Uncertainty
Date: 11/7/2020

Description: Simulates simplified version of Plants versus Zombies in a small 3x3 world with random planting policy.

This version takes place in a 3x3 world where the
home is on the left side of the grid and zombies enter on the right side of the grid. If a zombie enters the home, then
it eats your brains and you lose :( If you survive the night (20 time steps) then you win! Traditionally, the
player plants defensive plants that have the ability to kill the zombies in that row. Planting and zombie behavior are
modeled stochastically here to simulate a random planting policy.
"""

import pandas as pd
import numpy as np
import copy


def containsZombie(state):
    return True if 10 in state or 11 in state or 20 in state else False


def loosingState(state):
    home = [state[0], state[3], state[6]]
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
    landing = [state[2], state[5], state[8]]
    free = [j for j, k in enumerate(landing) if k == 0 or k == 3]
    if len(free) == 0:
        return eaten
    else:
        index = np.random.choice(free)
        if state[2 + (3*index)] == 3:
            # zombie ran into plant and ate it
            eaten += 1
        # add strong zombie with prob 0.5 and weak zombie with prob 0.5
        state[2 + (3*index)] = np.random.choice([10, 20], p=[0.5, 0.5])

    return eaten


def killZombies(state):
    killed = 0
    if not containsZombie(state):
        return killed

    for i in range(0, 7, 3):
        row = state[i:i+3]
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
    state = [0 for i in range(9)]
    # starting state always has one zombie
    addZombie(state)
    win = 0

    #generates (state, action, reward, sp)
    step = 0
    while True:
        state_curr = copy.copy(state)
        tup = [state_curr]

        # ACTION
        free = [j for j, k in enumerate(state) if k == 0]
        # if board is full, no choice but to wait
        if len(free) == 0:
            action = 9
        else :
            index = np.random.choice(free)
            # give the humans a chance, let them always plant plant at first timestep
            if step == 0:
                state[index] = 3
            # plant plant with prob 0.5 at each following timestep
            else:
                state[index] = np.random.choice([3, 0], p=[0.5, 0.5])
            # action 9 corresponds to not planting aka placing 0
            action = index if state[index] == 3 else 9
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


def generateData():
    data = pd.DataFrame(columns = ['s', 'a', 'r', 'sp'])
    win_count = 0
    for i in range(100):
        game_tup = simulateGame()
        data = data.append(game_tup[0])
        win_count += game_tup[1]
    print(win_count)
    return data


def main():
    data = generateData()
    #data.to_csv("game_data1.csv", index=False)


if __name__ == "__main__":
    main()

