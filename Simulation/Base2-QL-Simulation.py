import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Simulating dataset as rows of states for simplicity
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

# Initialize Q-table for Q-learning
q_table = np.zeros((len(dataset), len(actions)))

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate (epsilon-greedy strategy)

# Visualization logs
q_value_logs = {action: [] for action in actions}

# Function to simulate state transitions based on action
def simulate_transition(state, action):
    new_state = state.copy()

    if action == "Increase Study Hours":
        if state["Motivation"] == "Low":
            if random.random() < 0.3:  # Low motivation has lower probability of effective studying
                new_state["Hours_Studied"] += random.randint(1, 2)
                new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(1, 5), 100)
            else:
                new_state["Motivation"] = "Low"
        elif state["Motivation"] == "Medium":
            if random.random() < 0.7:
                new_state["Hours_Studied"] += random.randint(2, 4)
                new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 10), 100)
        elif state["Motivation"] == "High":
            new_state["Hours_Studied"] += random.randint(3, 5)
            new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(10, 15), 100)

    elif action == "Provide Tutoring":
        if state["Motivation"] == "Low" or state["Exam_Score"] < 70:
            if random.random() < 0.9:
                new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(10, 20), 100)
                new_state["Motivation"] = "Medium" if state["Motivation"] == "Low" else "High"
        elif state["Exam_Score"] >= 70:
            if random.random() < 0.7:
                new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 10), 100)

    elif action == "Introduce Motivational Feedback":
        if state["Motivation"] == "Low":
            if random.random() < 0.9:
                new_state["Motivation"] = "Medium"
            if random.random() < 0.5:
                new_state["Exam_Score"] = min(new_state["Exam_Score"] + 5, 100)
        elif state["Motivation"] == "Medium":
            if random.random() < 0.8:
                new_state["Motivation"] = "High"
            if random.random() < 0.6:
                new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 10), 100)

    elif action == "Improve Attendance":
        if state["Attendance"] < 75:
            if random.random() < 0.8:
                new_state["Attendance"] = min(new_state["Attendance"] + random.randint(10, 15), 100)
                new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(10, 15), 100)

    return new_state

# Function to calculate reward
def calculate_reward(state, action, new_state):
    reward = 0
    if new_state["Exam_Score"] > state["Exam_Score"]:
        reward += 10
    if new_state["Motivation"] == "High" and state["Motivation"] != "High":
        reward += 5
    return reward


# Q-learning simulation
def q_learning_simulation(n_episodes=5000):
    global q_table, q_value_logs

    for episode in range(n_episodes):
        state_idx = random.randint(0, len(dataset) - 1)
        state = dataset[state_idx]

        while state["Exam_Score"] < 100:
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(len(actions))
            else:
                action_idx = np.argmax(q_table[state_idx, :])

            action = actions[action_idx]
            next_state = simulate_transition(state, action)
            reward = calculate_reward(state, action, next_state)

            next_state_idx = state_idx

            q_table[state_idx, action_idx] += alpha * (
                reward + gamma * np.max(q_table[next_state_idx, :]) - q_table[state_idx, action_idx]
            )

            state = next_state

        for action_idx, action in enumerate(actions):
            q_value_logs[action].append(np.mean(q_table[:, action_idx]))

# Run the Q-learning simulation
q_learning_simulation()

# Display the final Q-table
q_table_df = pd.DataFrame(q_table, index=[f"Student {i+1}" for i in range(len(dataset))], columns=actions)
print("Final Q-Table:")
print(q_table_df)

# Visualization of Q-value evolution
plt.figure(figsize=(12, 6))
for action, values in q_value_logs.items():
    plt.plot(range(len(values)), values, label=action)
plt.title("Q-Learning: Evolution of Q-Values Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Average Q-Value")
plt.legend()
plt.grid()
plt.show()

# Determine optimal actions for each student
optimal_actions = q_table_df.idxmax(axis=1)
print("\nOptimal Actions for Each Student:")
print(optimal_actions)

# Determine the optimal action for each student
optimal_actions = q_table_df.idxmax(axis=1)

# Count how often each action is the optimal one
dominance = optimal_actions.value_counts()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bars = dominance.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Q-Learning: Dominance of Optimal Actions Across Students", fontsize=14)
plt.xlabel("Optimal Actions", fontsize=12)
plt.ylabel("Number of Students", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Adding percentage labels above each bar
for i in bars.patches:
    percentage = f"{(i.get_height() / len(dataset)) * 100:.1f}%"
    plt.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.5, percentage,
             ha="center", fontsize=10, color="black")

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

