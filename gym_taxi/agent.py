import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6,epsilon=0.,alpha=0.1,gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.ep=epsilon
        self.alpha=alpha
        self.gamma=gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs=np.ones(self.nA)*self.ep/self.nA
        probs[np.argmax(self.Q[state])]=1-self.ep+self.ep/self.nA
        return np.random.choice(self.nA,p=probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        max_q=self.Q[next_state].max()
        self.Q[state][action]=self.Q[state][action]+self.alpha*(reward+self.gamma*max_q-self.Q[state][action])