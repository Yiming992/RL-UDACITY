import sys
import gym
import numpy as np 
from collections import defaultdict
from plot_utils import plot_blackjack_values,plot_policy

def generate_episode_from_limit_stochastic(bj_env):
	episode=[]
	state=bj_env.reset()
	while True:
		probs=[0.8,0.2] if state[0]>18 else [0.2,0.8]
		action=np.random.choice(np.arange(2),p=probs)
		next_state,reward,done,info=bj_env.step(action)
		episode.append((state,action,reward))
		state=next_state
		if done:
			break
	return episode 

def mc_prediction_q(env,num_episodes,generate_episode,gamma=1.0,method='f'):
	returns_sum=defaultdict(lambda:np.zeros(env.action_space.n))
	N=defaultdict(lambda:np.zeros(env.action_space.n))
	Q=defaultdict(lambda:np.zeros(env.action_space.n))
	for i_episode in range(1,num_episodes+1):
		first=defaultdict(lambda:False)
		if i_episode%1000==0:
			print('\rEpisode {}/{}'.format(i_episode,num_episodes))
			sys.stdout.flush()
		episode=generate_episode(env)
		for index,step in enumerate(episode):
			state,action,reward=step
			if first[(state,action)]:
				continue
			first[(state,action)]=True
			for i in range(index,len(episode)):
				returns_sum[state][action]+=(gamma**(i-index))*episode[i][2]
	for state,value in returns_sum.items():
		Q[state]=value/N[state]
	return Q



def gen_episode(policy,env):
    episode=[]
    state=env.reset()
    while True:
        action=policy[state]
        next_state,reward,done,info=env.step(action)
        episode.append((state,action,reward))
        state=next_state
        if done:
            break
    return episode

def policy_update(policy,Q,env,num_episode):
    epsilon=1.0
    if num_episode%10000==0 and num_episode>0:
        epsilon=epsilon/(num_episode/5000)
    n=env.action_space.n
    action=np.arange(n)
    for state in Q.keys():
        prob=[]
        if Q[state].max()==Q[state].min():
            prob.append(0.5)
            prob.append(0.5)
            continue
        for a in action: 
            if Q[state][a]==Q[state].max():
                prob.append((1-epsilon)+epsilon/n)
            else:
                prob.append(epsilon/n)
        policy[state]=np.random.choice(action,p=prob)
    return policy


def mc_control(env,num_episodes,alpha,gamma=1.0):
    policy=defaultdict(lambda:0)
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        first_seen=defaultdict(lambda:True)
        # monitor progress
        episode=gen_episode(policy,env)
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        for index,step in enumerate(episode):
            state,action,reward=step
            if first_seen[(state,action)]:
                ac_value=0
                for i in range(index,len(episode)):
                    ac_value+=(gamma**(i-index))*episode[i][-1]
            else:
                continue
            Q[state][action]+=alpha*(ac_value-Q[state][action])
            first_seen[(state,action)]=False
        policy=policy_update(policy,Q,env,i_episode)
    return policy, Q