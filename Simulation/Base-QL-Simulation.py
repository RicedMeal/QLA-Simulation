import numpy as np
import random
import pandas as pd

# Load dataset (replace with your dataset handling logic)
# Simulating dataset as rows of states for simplicity
dataset = [
    {"Hours_Studied": 23, "Attendance": 84, "Motivation": "Low", "Exam_Score": 67},
    {"Hours_Studied": 19, "Attendance": 64, "Motivation": "Low", "Exam_Score": 59},
    {"Hours_Studied": 24, "Attendance": 89, "Motivation": "Medium", "Exam_Score": 74},
    {"Hours_Studied": 29, "Attendance": 89, "Motivation": "Medium", "Exam_Score": 71},
    {"Hours_Studied": 19, "Attendance": 92, "Motivation": "Medium", "Exam_Score": 70},
]

# Actions available to the agent
actions = ["Increase Study Hours", "Provide Tutoring", "Introduce Motivational Feedback"]

# Initialize Q-table (states x actions)
# Each state is represented by an index in the dataset
q_table = np.zeros((len(dataset), len(actions)))

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate (epsilon-greedy strategy)

# Function to simulate state transitions based on action
def simulate_transition(state, action):
    """Define your own rules here for state transitions."""
    # Should be backed with empirical studies
    new_state = state.copy()
    
    # Example Rules (you can modify these based on your logic)
    if action == "Increase Study Hours":
        # Rule: Increasing study hours improves exam score with some probability
        if random.random() < 0.7:  # 70% chance of success
            new_state["Hours_Studied"] += 2
            new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 10), 100)
    elif action == "Provide Tutoring":
        # Rule: Tutoring improves exam score and motivation
        if random.random() < 0.8:  # 80% chance of success
            new_state["Exam_Score"] = min(new_state["Exam_Score"] + random.randint(5, 15), 100)
            new_state["Motivation"] = "Medium" if state["Motivation"] == "Low" else "High"
    elif action == "Introduce Motivational Feedback":
        # Rule: Motivational feedback increases engagement but not directly scores
        if random.random() < 0.9:  # 90% chance of success
            new_state["Motivation"] = "High"
    
    return new_state

# Function to calculate reward
def calculate_reward(state, action, new_state):
    """Define your own reward system."""
    reward = 0
    
    # Example Reward Logic
    if new_state["Exam_Score"] > state["Exam_Score"]:
        reward += 10  # Reward for score improvement
    if new_state["Motivation"] == "High" and state["Motivation"] != "High":
        reward += 5  # Reward for motivation improvement
    
    return reward

# Q-learning simulation
def q_learning_simulation(episodes=100):
    global q_table
    
    for episode in range(episodes):
        # Start with a random state
        current_state_idx = random.randint(0, len(dataset) - 1)
        current_state = dataset[current_state_idx]
        
        while True:
            # Choose an action (epsilon-greedy)
            if random.random() < epsilon:
                action_idx = random.randint(0, len(actions) - 1)  # Explore
            else:
                action_idx = np.argmax(q_table[current_state_idx])  # Exploit
            
            action = actions[action_idx]
            
            # Simulate the action's effect
            new_state = simulate_transition(current_state, action)
            
            # Calculate reward
            reward = calculate_reward(current_state, action, new_state)
            
            # Find the new state's index (assume deterministic mapping for simplicity)
            new_state_idx = current_state_idx  # Modify if dynamic states are added
            
            # Update Q-value using the Bellman equation
            q_table[current_state_idx, action_idx] = q_table[current_state_idx, action_idx] + alpha * (
                reward + gamma * np.max(q_table[new_state_idx]) - q_table[current_state_idx, action_idx]
            )
            
            # Transition to new state
            current_state = new_state
            
            # Break if terminal condition (optional: e.g., max score)
            if current_state["Exam_Score"] >= 100:
                break

# Run simulation
q_learning_simulation(episodes=500)

# Display final Q-table as a labeled DataFrame
q_table_df = pd.DataFrame(q_table, index=[f"Student {i+1}" for i in range(len(dataset))], columns=actions)

print("Final Q-Table with Labels:")
print(q_table_df)

# Explain optimal actions for each student
optimal_actions = q_table_df.idxmax(axis=1)  # Get the action with the highest Q-value for each student

print("\nOptimal Actions for Each Student:")
print(optimal_actions)

# Legend for Q-values
print("\nLegend for Q-Values:")
print("""
Q-Value Range   Meaning
50+             Highly effective action for the student.
30–50           Moderately effective action. 
10–30           Limited benefit. Might not maximize the student’s outcomes.
0–10            Likely ineffective or counterproductive for the student's current state.
""")
