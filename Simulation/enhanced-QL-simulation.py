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
]

actions = ["Increase Study Hours", "Provide Tutoring", "Introduce Motivational Feedback", "Improve Attendance", ]

# Initialize Q-table (states x actions)
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
        if random.random() < 0.7:
            new_state["Hours_Studied"] += 2
            new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 10), 100)
    elif action == "Provide Tutoring":
        if random.random() < 0.8:
            new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 15), 100)
            new_state["Motivation"] = "Medium" if state["Motivation"] == "Low" else "High"
    elif action == "Introduce Motivational Feedback":
        if random.random() < 0.9:
            new_state["Motivation"] = "High"
    elif action == "Improve Attendance":
        if random.random() < 0.6:
            new_state["Attendance"] = min(new_state["Attendance"] + random.randint(5, 15), 100)
            new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 10), 100)
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
def q_learning_simulation(episodes=500):
    global q_table, q_value_logs

    # Reset Q-value logs to avoid accumulation
    q_value_logs = {action: [] for action in actions}
    
    for episode in range(episodes):
        current_state_idx = random.randint(0, len(dataset) - 1)
        current_state = dataset[current_state_idx]
        while True:
            if random.random() < epsilon:
                action_idx = random.randint(0, len(actions) - 1)
            else:
                action_idx = np.argmax(q_table[current_state_idx])
            
            action = actions[action_idx]
            new_state = simulate_transition(current_state, action)
            reward = calculate_reward(current_state, action, new_state)
            new_state_idx = current_state_idx

            # Log Q-value updates for visualization
            old_q_value = q_table[current_state_idx, action_idx]
            future_max_q = np.max(q_table[new_state_idx])
            updated_q_value = old_q_value + alpha * (reward + gamma * future_max_q - old_q_value)
            q_table[current_state_idx, action_idx] = updated_q_value

            # Log for visualization
            q_value_logs[action].append(updated_q_value)

            current_state = new_state
            if current_state["Exam_Score"] >= 100:
                break

# Run simulation
q_learning_simulation(episodes=500)

# Display final Q-table
q_table_df = pd.DataFrame(q_table, index=[f"Student {i+1}" for i in range(len(dataset))], columns=actions)
print("Final Q-Table with Labels:")
print(q_table_df)

# Explain optimal actions for each student
optimal_actions = q_table_df.idxmax(axis=1)
print("\nOptimal Actions for Each Student:")
print(optimal_actions)

# Refined Visualization for Q-value evolution (Overestimation Bias)
plt.figure(figsize=(12, 6))
for action, values in q_value_logs.items():
    plt.plot(values, label=action)
plt.title("Overestimation: Evolution of Q-Values Over Episodes", fontsize=14)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Q-Value", fontsize=12)
plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label="Initial Growth Phase")
plt.axvline(x=300, color='orange', linestyle='--', alpha=0.7, label="Saturation Phase")
plt.legend(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Refined Visualization for Dominance (Overfitting)
dominance = q_table_df.idxmax(axis=1).value_counts()

plt.figure(figsize=(10, 6))
bars = dominance.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Overfitting: Dominance of Optimal Actions Across Students", fontsize=14)
plt.xlabel("Optimal Actions", fontsize=12)
plt.ylabel("Number of Students", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
for i in bars.patches:
    percentage = f"{(i.get_height() / len(dataset)) * 100:.1f}%"
    plt.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.5, percentage, ha="center", fontsize=10, color="black")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


