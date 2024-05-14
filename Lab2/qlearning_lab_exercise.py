import numpy as np
import environment as env
#import math

# actual state space 
# (1,1),(2,1),(3,1),(4,1),(1,2),(3,2),(4,2),(1,3),(2,3),(3,3),(4,3)
#   0     1     2     3     4     5     6     7     8     9    10

# For coding simplicity, use the below one. 
# (1,1),(2,1),(3,1),(4,1),(1,2),(2,2),(3,2),(4,2),(1,3),(2,3),(3,3),(4,3)
#   0     1     2     3     4     5     6     7     8     9    10    11

# action space
# N,S,E,W
# 0,1,2,3

np.random.seed(seed=1) # with 1000 iterations, I have a decent solution. 
#np.random.seed(seed=2)
#np.random.seed(seed=50)

# variables/hyperparameters 
size_state_space = 12 # for simplicity, let us use 12 states, instead of 11. 
size_action_space = 4 # 0:N, 1:S, 2:E, 3:W
n_episodes = 1000
factor_discount = 1.
#prob_transition = [0.8,0.1,0.1] # front, left, right 
eps = 0.1 # for epsilon greedy
alpha = 0.1 # learning rate


# initial action value function
Q = np.zeros((size_state_space,size_action_space))

#############################################################################
## Start your lab exercise from here
## To get (s_{t+1}, R_t) from (s_t, a_t) use env.step() function from environment.py, 
## where both s_t and s_{t+1} follow the enumeration system that goes from 0 to 11. 

for episode in range(n_episodes):

    t = 0
    state = np.random.randint(size_state_space)
    while True:
        m = np.random.rand()
        # Exploration
        if m < eps:
            action = np.random.randint(size_action_space)
        # Exploitation
        else:
            action = np.argmax(Q[state])
        
        reward, next_state = env.step(state, action)

        # Update the Q-value using Q-learning
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + factor_discount * np.max(Q[next_state]))
        state = next_state
        t += 1
        if state in [7,11]:  
            break



#############################################################################





# search for the optimal policy
#policy = np.argmax(value_future,axis=1)
policy = np.argmax(Q,axis=1)
policy[5] = -1
policy[7] = -1
policy[11] = -1


policyl = np.empty((size_state_space,1),dtype=str)
policyl[policy==0]='N'
policyl[policy==1]='S'
policyl[policy==2]='E'
policyl[policy==3]='W'
policyl[5] = '*'
policyl[7] = '*'
policyl[11] = '*'


policy2d = np.reshape(policy,(3,4))
policy2d = np.flip(policy2d,axis=0)

policy2dl = np.reshape(policyl,(3,4))
policy2dl = np.flip(policy2dl,axis=0)

print('Q-table:\n',Q)
print('Policy:\n',policy2dl)
