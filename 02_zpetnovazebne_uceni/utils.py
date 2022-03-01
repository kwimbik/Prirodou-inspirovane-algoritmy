import gym
import numpy as np
import utils
import matplotlib.pyplot as plt

class StateDiscretizer:
  # predani rozmeru prostredi a spojitych stavu a jejich rozdeleni na diskretni intervaly
    def __init__(self, ranges, states):
        pass
    
    # prirazeni stavu do spravneho intervalu
    def transform(self, obs):
        pass
        
class QLearningAgent:
    # nastaveni moznych akci - L, N, R   
    # diskretizace stavu prostredi
    # definice matice uzitku Q[stavy, akce]
    # promenna na zapamatovani si minuleho stavu a minule akce
    # donastaveni dalsich parametru trenovani
    def __init__(self, actions, state_transformer, train=True):
        pass
    
    # na zaklade stavu a akce se vybira nova akce
    # 1. najde se nejlepsi akce pro dany stav
    # 2. s malou pravd. vezme nahodnou
    # 3. updatuje se Q matice
    def act(self, observe, reward, done):
        pass

    # reset minuleho stavu a akce na konci epizody
    def reset(self):
        pass



env = gym.make('MountainCar-v0')
SD = StateDiscretizer(env.observation_space, [])
agent = QLearningAgent(env.action_space, SD)
total_rewards = []
for i in range(1000):
    obs = env.reset()
    agent.reset()    
    done = False
    
    r = 0
    R = 0 # celkova odmena - jen pro logovani
    t = 0 # cislo kroku - jen pro logovani
    
    while not t < 1000:
        action = agent.act(obs, r, done)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
        
total_rewards.append(R)
env.close()

def show_animation(agent, env, steps=200, episodes=1):
    ''' Pomocna funkce, ktera zobrazuje chovani zvoleneho agenta v danem 
    prostredi.
    Parameters
    ----------
    agent: 
        Agent, ktery se ma vizualizivat, musi implementovat metodu
        act(observation, reward, done)
    env:
        OpenAI gym prostredi, ktere se ma pouzit
    
    steps: int
        Pocet kroku v prostredi, ktere se maji simulovat
    
    episodes: int
        Pocet episod, ktere se maji simulovat - kazda a pocet kroku `steps`.
    '''
    for i in range(episodes):
        obs = env.reset()
        done = False
        R = 0
        t = 0
        r = 0
        while not done and t < steps:
            env.render()
            action = agent.act(obs, r, done)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
        agent.reset()


def moving_average(x, n):
    weights = np.ones(n)/n
    return np.convolve(np.asarray(x), weights, mode='valid')