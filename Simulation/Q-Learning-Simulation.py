#This simulation is based from this: Authors: I Amzil, S Aammou, Z Tagdimi, et al.
#Published: 2024
#Link: https://www.atlantis-press.com/proceedings/elses-23/125997585
"""Summary: This study explores the use of Q-learning algorithms in 
adaptive educational systems to improve personalization and learning outcomes. 
It suggests that adaptive educational systems informed by Q-learning algorithms can enhance the learning experience."""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Q-learning parameters
n_states = 4  # number of states
n_actions = 3  # number of actions (e.g., easy, medium, hard)
epsilon = 0.1  # exploration rate
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor

# Initialize the Q-values to zeros
Q = np.zeros((n_states, n_actions))

# Define the e-learning environment
def learning_env(state, action):
    if state == 0:  # Starting a new learning session
        if action == 0:  # Choose an easy question
            next_state = 1  # Assume they pass
            reward = 1
        elif action == 1:  # Choose a medium question
            next_state = 2  # Assume they fail
            reward = -1
        else:  # Choose a hard question
            next_state = 2  # Assume they fail
            reward = -1
    elif state == 1:  # Answered an activity and passed
        if action == 0:  # Choose an easy question
            next_state = 1  # Stay in the same state
            reward = 1
        elif action == 1:  # Choose a medium question
            next_state = 1  # Stay in the same state
            reward = 1
        else:  # Choose a hard question
            next_state = 3  # Move to re-learn state
            reward = -1
    elif state == 2:  # Answered an activity and failed
        if action == 0:  # Choose an easy question
            next_state = 1  # Move to pass state
            reward = 1
        elif action == 1:  # Choose a medium question
            next_state = 2  # Stay in the same state
            reward = -1
        else:  # Choose a hard question
            next_state = 3  # Move to re-learn state
            reward = -1
    else:  # State 3: Re-learn the lesson (terminal state)
        next_state = state
        reward = 0  # No reward for terminal state

    return next_state, reward

# Q-Learning algorithm
n_episodes = 1000  # number of episodes for training
rewards_per_episode = []  # Track total rewards per episode
exploration_count = 0  # Track exploration actions
exploitation_count = 0  # Track exploitation actions

for episode in range(n_episodes):
    total_reward = 0  # Track total rewards in this episode
    state = 0  # Start from the initial state
    while state < 3:  # Continue until a terminal state is reached
        # Choose an action using epsilon-greedy exploration strategy
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
            exploration_count += 1  # Increment exploration count
        else:
            action = np.argmax(Q[state, :])
            exploitation_count += 1  # Increment exploitation count
        # Take the chosen action and observe the next state and reward
        next_state, reward = learning_env(state, action)
        total_reward += reward  # Accumulate reward
        # Update the Q-value for the current state-action pair
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        # Move to the next state
        state = next_state
    rewards_per_episode.append(total_reward)  # Record total rewards for this episode

# Output Results
print("\nQ-Learning Algorithm for Adaptive Educational Systems Simulation\n")
print("Q-Table Values:")
print(Q)

print("\nConvergence Metrics:")
print(f"Average change in Q-values in the last 100 episodes: {np.mean(np.abs(np.diff(Q, axis=0))[-100:])}")

print("\nReward Patterns:")
print(f"Average reward per episode: {np.mean(rewards_per_episode)}")
print(f"Total rewards over all episodes: {np.sum(rewards_per_episode)}")

print("\nExploration vs. Exploitation:")
print(f"Exploration actions taken: {exploration_count}")
print(f"Exploitation actions taken: {exploitation_count}")

print("\nLearning Parameters:")
print(f"Learning rate (alpha): {alpha}")
print(f"Discount factor (gamma): {gamma}")
print(f"Exploration rate (epsilon): {epsilon}")

# Visualization 1: Exploration vs Exploitation
plt.figure(figsize=(8, 6))
plt.bar(['Exploration', 'Exploitation'], [exploration_count, exploitation_count], color=['blue', 'green'])
plt.title('Exploration vs. Exploitation Actions')
plt.ylabel('Number of Actions Taken')
plt.show()

# Visualization 2: Q-Table Heatmap (Overestimation of Learning Strategies)
plt.figure(figsize=(8, 6))
sns.heatmap(Q, annot=True, cmap='coolwarm', cbar=True, xticklabels=['Easy', 'Medium', 'Hard'], yticklabels=['Start', 'Passed', 'Failed', 'Re-learn'])
plt.title('Q-Table Heatmap: Learning Strategies')
plt.xlabel('Actions')
plt.ylabel('States')
plt.show()

# Visualization 3: Rewards Over Episodes (Limited Adaptability)
plt.figure(figsize=(10, 6))
plt.plot(range(len(rewards_per_episode)), rewards_per_episode, color='orange', label='Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Total Rewards Over Time')
plt.legend()
plt.grid(True)
plt.show()
