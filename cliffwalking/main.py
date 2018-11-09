import numpy as np
from collections import defaultdict



def epsilon_greedy(env,state,epsilon,Q):
	probs=np.ones(env.nA)*epsilon/env.nA
	max_index=np.argmax(Q[state])
	probs[max_index]=1-epsilon+probs[max_index]
	action=np.random.choice(np.arange(4),p=probs)
	return action,probs

# TD Control Sarsa

def sarsa(env,num_episodes,alpha,gamma=1.0):
	Q=defaultdict(lambda:np.zeros(env.nA))
	for i in range(1,num_episode+1):
		if i%100==0:
			print("\rEpisode {}/{}".format(i_episode,num_episodes),end=' ')
			sys.stdout.flush()
		current_action,_=epsilon_greedy(env,prev_state,0,Q)
		state,rerward,done,info=env.step(current_action)
		while not done:
			action,_=epsilon_greedy(env,state,0,Q)
			Q[prev_state][current_action]=Q[prev_state][current_action]+alpha*(reward+gamma*(Q[state][action])-Q[prev_state][current_action])
			prev_state=state
			current_action=action 
			state,reward,done,info=env.step(action)
	return Q



# TD Control Q-Learning

def q_learning(env,num_episodes,alpha,gamma=1.0):
    Q=defaultdict(lambda:np.zeros(env.nA))
    for i in range(1,num_episodes+1):
        prev_state=env.reset()
        if i %100==0:
            print("\rEpisode {}/{}".format(i,num_episodes),end=" ")
            sys.stdout.flush()
        current_action,_=epsilon_greedy(env,prev_state,0,Q)
        state,reward,done,info=env.step(current_action)
        t=0
        while not done and t<300:
#             future_state,_,done,info=env.step(action)
            Q[prev_state][current_action]=Q[prev_state][current_action]+alpha*(reward+gamma*(Q[state].max())-Q[prev_state][current_action])
            t+=1
            prev_state=state
            action,_=epsilon_greedy(env,state,0,Q)
            current_action=action
            state,reward,done,info=env.step(action)           
    return Q


# TD Control Expected Sarsa

def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        prev_state=env.reset()
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        current_action,probs=epsilon_greedy(env,prev_state,0,Q)
        state,reward,done,info=env.step(current_action)
        t=0
        while not done and t<300:
            action,probs=epsilon_greedy(env,state,0,Q)
            Q[prev_state][current_action]=Q[prev_state][current_action]+alpha*(reward+gamma*(Q[state]*probs).sum()-Q[prev_state][current_action])
            prev_state=state
            current_action=action
            state,rerward,done,info=env.step(action)
    return Q