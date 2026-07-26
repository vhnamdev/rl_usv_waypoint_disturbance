# USV Multi-Waypoint Control with Reinforcement Learning

This project applies Reinforcement Learning to the autonomous control of an Unmanned Surface Vehicle using Python, PyTorch, Pygame, and Matplotlib.

The USV acts as an agent. It learns how to move through multiple waypoints in the correct order while handling water currents, wind, waves, and ocean boundaries.

## Demo

![USV RL Demo](demo.gif)

## Reinforcement Learning Structure

- **Agent:** The Unmanned Surface Vehicle
- **Environment:** A two-dimensional ocean simulation
- **State:** Waypoint position, heading error, and USV velocities
- **Actions:** Surge-force and yaw-moment commands
- **Reward:** Feedback returned after each movement

The state contains 8 values:

- Relative waypoint position along the X-axis
- Relative waypoint position along the Y-axis
- Distance to the current waypoint
- Sine and cosine of the heading error
- Surge velocity
- Sway velocity
- Yaw rate

The neural network receives these 8 state values and predicts Q-values for five possible actions:

```text
Action 0 → Coast
Action 1 → Move forward slowly
Action 2 → Move forward quickly
Action 3 → Move forward and turn left
Action 4 → Move forward and turn right
```

## Reward System

```text
Move closer to waypoint:     Positive reward
Move away from waypoint:     Negative reward
Reach one waypoint:          +100
Complete all waypoints:      +300
Leave the ocean boundary:    -200
Reach the maximum steps:     -100
```

A small penalty is also applied for taking too long and using unnecessary control force.

## Environmental Disturbances

The USV is affected by:

- Water currents
- Wind forces
- Wave moments
- Random wind gusts

The disturbances change during training so the agent learns to control the USV under different environmental conditions.

## Learning Method

The agent is trained using Deep Q-Learning with:

- Epsilon-greedy exploration
- Experience replay
- Short-memory training
- Long-memory training
- Bellman Q-value updates
- Main and target neural networks
- Huber loss
- Adam optimizer
- Gradient clipping
- Best-model saving
- Training checkpoint saving

At the beginning, the USV performs more random actions to explore the environment. After more training episodes, it gradually relies on the neural network and selects actions with higher predicted Q-values.

## Training Visualization

Pygame displays:

- The USV position and heading
- Current and future waypoints
- Completed waypoints
- The reference path
- USV velocity information

Matplotlib displays:

- Reward for each episode
- Mean reward
- Number of completed waypoints

## Technologies

- Python
- PyTorch
- Pygame
- NumPy
- Matplotlib

## Project Structure

```text
rl_usv_waypoint/
├── agent.py
├── config.py
├── disturbance.py
├── evaluate.py
├── helper.py
├── model.py
├── rl_environment.py
├── train.py
├── usv_dynamics.py
├── usv_environment.py
├── demo.gif
├── checkpoints/
│   ├── best_model.pth
│   └── training_checkpoint.pth
└── README.md
```

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install torch pygame numpy matplotlib
```

On Ubuntu, Matplotlib may require Tkinter:

```bash
sudo apt install python3-tk
```

## Train the Model

Start the training process:

```bash
python train.py
```

The program will display the USV simulation, episode reward, mean reward, completed waypoints, epsilon, and training loss.

## Evaluate the Model

Run the trained model without random exploration:

```bash
python evaluate.py
```

The trained USV will attempt to complete all generated waypoints without further model training.

## Purpose

The purpose of this project is to understand how Reinforcement Learning can be applied to an autonomous vehicle control problem:

```text
State
→ Action
→ USV Dynamics
→ Reward
→ Next State
→ Model Training
```

This project provides a basic foundation for applying Reinforcement Learning to more advanced autonomous marine vehicle control systems in the future.