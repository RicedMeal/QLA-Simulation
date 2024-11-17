import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Define dataset (states) and actions
dataset = [
    {"Hours_Studied": 23, "Attendance": 84, "Motivation": "Low", "Exam_Score": 67},
    {"Hours_Studied": 19, "Attendance": 64, "Motivation": "Low", "Exam_Score": 59},
    {"Hours_Studied": 24, "Attendance": 89, "Motivation": "Medium", "Exam_Score": 74},
    {"Hours_Studied": 29, "Attendance": 89, "Motivation": "Medium", "Exam_Score": 71},
    {"Hours_Studied": 19, "Attendance": 92, "Motivation": "Medium", "Exam_Score": 70},
    {"Hours_Studied": 19, "Attendance": 88, "Motivation": "Medium", "Exam_Score": 71},
    {"Hours_Studied": 29, "Attendance": 84, "Motivation": "Low", "Exam_Score": 67},
    {"Hours_Studied": 25, "Attendance": 78, "Motivation": "Medium", "Exam_Score": 66},
    {"Hours_Studied": 17, "Attendance": 94, "Motivation": "High", "Exam_Score": 69},
    {"Hours_Studied": 23, "Attendance": 98, "Motivation": "Medium", "Exam_Score": 72},
    {"Hours_Studied": 17, "Attendance": 80, "Motivation": "Medium", "Exam_Score": 68},
    {"Hours_Studied": 17, "Attendance": 97, "Motivation": "Low", "Exam_Score": 71},
    {"Hours_Studied": 21, "Attendance": 83, "Motivation": "Low", "Exam_Score": 70},
    {"Hours_Studied": 9, "Attendance": 82, "Motivation": "Medium", "Exam_Score": 66},
    {"Hours_Studied": 10, "Attendance": 78, "Motivation": "Medium", "Exam_Score": 65},
    {"Hours_Studied": 17, "Attendance": 68, "Motivation": "Medium", "Exam_Score": 64},
    {"Hours_Studied": 14, "Attendance": 60, "Motivation": "Low", "Exam_Score": 60},
    {"Hours_Studied": 22, "Attendance": 70, "Motivation": "Medium", "Exam_Score": 65},
    {"Hours_Studied": 15, "Attendance": 80, "Motivation": "Low", "Exam_Score": 67},
    {"Hours_Studied": 12, "Attendance": 75, "Motivation": "Medium", "Exam_Score": 66}
]


actions = ["Increase Study Hours", "Provide Tutoring", "Introduce Motivational Feedback", "Improve Attendance"]

# Initialize Q-table
q_table = np.zeros((len(dataset), len(actions)))

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Priority queue for transitions (state_idx, action_idx, reward, next_state_idx, TD-error)
priority_queue = []

# Function to simulate state transitions based on action
def simulate_transition(state, action):
    new_state = state.copy()

    if action == "Increase Study Hours":
        if state["Motivation"] == "Low":
            new_state["Exam_Score"] += random.randint(0, 5)
            new_state["Motivation"] = "Medium" if random.random() < 0.5 else "Low"
        elif state["Motivation"] == "Medium":
            new_state["Exam_Score"] += random.randint(5, 10)
            new_state["Motivation"] = "High" if random.random() < 0.3 else "Medium"
        elif state["Motivation"] == "High":
            new_state["Exam_Score"] += random.randint(10, 15)

    elif action == "Provide Tutoring":
        new_state["Exam_Score"] += random.randint(5, 20)
        new_state["Motivation"] = "High"

    elif action == "Introduce Motivational Feedback":
        if state["Motivation"] == "Low":
            new_state["Motivation"] = "Medium"
            new_state["Exam_Score"] += random.randint(0, 5)
        elif state["Motivation"] == "Medium":
            new_state["Motivation"] = "High"
            new_state["Exam_Score"] += random.randint(5, 10)
        elif state["Motivation"] == "High":
            new_state["Exam_Score"] += random.randint(2, 5)

    elif action == "Improve Attendance":
        if state["Attendance"] < 90:
            new_state["Attendance"] += random.randint(5, 10)
            new_state["Exam_Score"] += random.randint(5, 10)

    new_state["Exam_Score"] = min(new_state["Exam_Score"], 100)  # Cap the score at 100
    return new_state

# Function to calculate reward
def calculate_reward(state, action, new_state):
    reward = 0
    if new_state["Exam_Score"] > state["Exam_Score"]:
        reward += 10
    if new_state["Motivation"] == "High" and state["Motivation"] != "High":
        reward += 5
    if new_state["Attendance"] > state["Attendance"]:
        reward += 3
    return reward

# Function to compute TD-error
def compute_td_error(state_idx, action_idx, reward, next_state_idx):
    return abs(reward + gamma * np.max(q_table[next_state_idx, :]) - q_table[state_idx, action_idx])

# Function to sample transitions based on priority
def sample_with_priority(priority_queue, batch_size):
    priorities = [transition[4] for transition in priority_queue]
    probabilities = priorities / np.sum(priorities)
    sampled_indices = np.random.choice(len(priority_queue), size=batch_size, p=probabilities)
    return [priority_queue[i] for i in sampled_indices]

# Initialize Q-value logs for visualization
q_value_logs = {action: [] for action in actions}

# Modified PER-enhanced Q-learning simulation
def prioritized_q_learning_with_logging(n_episodes=5000, batch_size=16):
    global q_table, priority_queue, q_value_logs

    for episode in range(n_episodes):
        # Start from a random initial state
        state_idx = random.randint(0, len(dataset) - 1)
        state = dataset[state_idx]

        while state["Exam_Score"] < 100:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(len(actions))
            else:
                action_idx = np.argmax(q_table[state_idx, :])

            action = actions[action_idx]
            next_state = simulate_transition(state, action)
            reward = calculate_reward(state, action, next_state)

            next_state_idx = state_idx  # Static dataset scenario

            # Calculate TD-error
            td_error = compute_td_error(state_idx, action_idx, reward, next_state_idx)

            # Update Q-value
            q_table[state_idx, action_idx] += alpha * (
                reward + gamma * np.max(q_table[next_state_idx, :]) - q_table[state_idx, action_idx]
            )

            # Add transition to priority queue
            priority_queue.append((state_idx, action_idx, reward, next_state_idx, td_error))

            # Limit the size of the priority queue
            if len(priority_queue) > 1000:
                priority_queue.pop(0)

            # Sample a batch of transitions from priority queue and update Q-values
            if len(priority_queue) >= batch_size:
                batch = sample_with_priority(priority_queue, batch_size)
                for (s_idx, a_idx, r, ns_idx, _) in batch:
                    td_error = compute_td_error(s_idx, a_idx, r, ns_idx)
                    q_table[s_idx, a_idx] += alpha * (
                        r + gamma * np.max(q_table[ns_idx, :]) - q_table[s_idx, a_idx]
                    )

                    # Update the TD-error in the priority queue
                    priority_queue = [
                        (s, a, rw, ns, compute_td_error(s, a, rw, ns)) if (s == s_idx and a == a_idx) else (s, a, rw, ns, _)
                        for (s, a, rw, ns, _) in priority_queue
                    ]

            # Update state
            state = next_state

        # Log average Q-values for each action at the end of the episode
        for action_idx, action in enumerate(actions):
            avg_q_value = np.mean(q_table[:, action_idx])
            q_value_logs[action].append(avg_q_value)

# Run the simulation with logging
prioritized_q_learning_with_logging()

# Display the final Q-table
q_table_df = pd.DataFrame(q_table, index=[f"Student {i+1}" for i in range(len(dataset))], columns=actions)
print("Final Q-Table:")
print(q_table_df)

# Visualize the dominance of optimal actions
optimal_actions = q_table_df.idxmax(axis=1)
dominance = optimal_actions.value_counts()

plt.figure(figsize=(10, 6))
bars = dominance.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Prioritized Q-Learning: Dominance of Optimal Actions")
plt.xlabel("Optimal Actions")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualize the evolution of Q-values over episodes
plt.figure(figsize=(12, 6))
for action, values in q_value_logs.items():
    plt.plot(range(len(values)), values, label=action)
plt.title("Evolution of Q-Values Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Average Q-Value")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
