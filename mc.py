#!/usr/bin/env python3
import gym
import matplotlib.pyplot as plt
import numpy as np

### LearningRate (=Alpha), Gamma (=Discount), Epsilon (=Randomness)

LR = 0.05 
GAMMA = 0.975
EPSILON = 0.99

EPISODES = 2500

START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES/3
EPSILON_END = 0.01

EPSILON_DECAY = (EPSILON-EPSILON_END)/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

env = gym.make('MountainCar-v0')

# Continuous to discrete
buckets = [20, 8]
discrete_size = (env.observation_space.high - env.observation_space.low)/buckets

q_table = np.random.uniform(low=-2, high=-1,
                            size=(buckets[0], buckets[1], env.action_space.n))

def observation2state(obs, env):
    nDiscreteState = (obs - env.observation_space.low)/discrete_size
    return tuple(nDiscreteState.astype(np.int))


rewardList = []
avgRewardList = []

# Episode loop
for currEpisode in range(EPISODES):
    if(currEpisode%1000 == 0 or currEpisode == EPISODES-1):
        print(currEpisode)
        render = True
    else:
        render = False

    totalReward = 0

    # X,Y to Q-table
    discreteState = observation2state(env.reset(), env)

    # agent loop
    done = False
    while not done:
        # epsilon greedy
        if(np.random.random() < EPSILON): # if smaller then EPS choose on random (=explore)
            action = np.random.randint(0, env.action_space.n)
        else: # otherwise choose by qtable
            action = np.argmax(q_table[discreteState])

        observation, reward, done, _ = env.step(action)
        if(render):
            env.render()

        newDiscreteState = observation2state(observation, env)

        if not done:
            # update q values

            # max possible Q value in future step
            maxQ = np.max(q_table[newDiscreteState])

            # current q value (already performed action)
            currQ = q_table[discreteState + (action,)]

            # newQ calculation
            newQ = (1-LR) * currQ + LR*(reward + GAMMA * maxQ)

            # quality value update
            q_table[discreteState + (action,)] = newQ

        # obs[0] = sim position
        elif(observation[0] >= env.goal_position):
            q_table[discreteState + (action,)] = 0

        # next state
        discreteState = newDiscreteState

        totalReward += reward

    rewardList.append(totalReward)

    if((currEpisode + 1) % (EPISODES//100) == 0):
        avgReward = np.mean(rewardList)
        avgRewardList.append(avgReward)
        rewardList = []

    # EPSILON decay
    if END_EPSILON_DECAYING >= currEpisode >= START_EPSILON_DECAYING:
            EPSILON -= EPSILON_DECAY


env.close()

plt.plot(np.arange(len(avgRewardList))+1,avgRewardList)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Reward vs Episodes')
plt.show()
