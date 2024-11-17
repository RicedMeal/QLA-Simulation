import numpy as np

# Define the Q-learning parameters
n_states = 5  # number of states
n_actions = 3  # number of actions (e.g., easy, medium, hard)
epsilon = 0.1  # exploration rate
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor

# Initialize the Q-values to zero
Q = np.zeros((n_states, n_actions))

# Define the environment
# Here, we assume a simplified environment with 5 states and 3 actions for demonstration purposes
# You can replace this with your own e-learning environment implementation
def transition(state, action):
    # Defining the transition function that maps from state-action pairs to next state and reward
    # For example, state 0 corresponds to an easy question, state 1 corresponds to a medium question, and so on
    # The action represents the difficulty level of the question: 0 for easy, 1 for medium, 2 for hard
    # The next state is determined based on the current state and the chosen action, and the reward is computed based on the correctness of the answer
    
    if state == 0:
        if action == 0: next_state = 1; reward = 1  # correct answer
        elif action == 1: next_state = 2; reward = -1  # incorrect answer
        else: next_state = 3; reward = -1  # incorrect answer
    elif state == 1:
        if action == 0: next_state = 3; reward = -1  # incorrect answer
        elif action == 1: next_state = 2; reward = 1  # correct answer
        else: next_state = 4; reward = -1  # incorrect answer
    elif state == 2:
        if action == 0: next_state = 4; reward = 1  # correct answer
        else: next_state = 4; reward = -1  # incorrect answer
    else:
        # State 3 and 4 are terminal states with no actions available
        next_state = state
        reward = 0  # no reward for terminal states
    
    return next_state, reward

# Q-learning algorithm
n_episodes = 1000  # number of episodes for training
for episode in range(n_episodes):
    state = 0  # start from the initial state
    
    # Continue until a terminal state is reached
    while state < 3:  # continuing until a terminal state is reached
        # Choose an action using epsilon-greedy exploration strategy
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state, :])
        
        # Take the chosen action and observe the next state and reward
        next_state, reward = transition(state, action)
        
        # Update the Q-value for the current state-action pair
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        
        # Move to the next state
        state = next_state

# After training, the Q-values represent the learned knowledge about the optimal actions in each state

# Demonstration of the learned policy
state = 0
while state < 3:  # demonstrating until a terminal state is reached
    action = np.argmax(Q[state, :])
    if action == 0:
        print("Selected action: Easy")
    elif action == 1:
        print("Selected action: Medium")
    else:
        print("Selected action: Hard")
    
    # Move to the next state
    next_state, _ = transition(state, action)
    state = next_state

    if state >= 3:
        print("Terminal state reached.")
