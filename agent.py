import os
import random
from collections import deque

import numpy as np
import torch

from model import LinearQNet
from model import QTrainer


# Maximum number of experiences stored in memory
MAX_MEMORY = 100_000

# Number of experiences used in one replay batch
BATCH_SIZE = 256

# Neural network learning rate
LEARNING_RATE = 0.001

# Future reward discount factor
GAMMA = 0.95

# Number of neurons in the first hidden layer
HIDDEN_SIZE = 256

# Initial exploration probability
EPSILON_START = 1.0

# Minimum exploration probability
EPSILON_END = 0.05

# Number of steps used to reduce epsilon
EPSILON_DECAY_STEPS = 50_000

# Number of training steps between target model updates
TARGET_UPDATE_FREQUENCY = 1000


class Agent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        # Number of values in one environment state
        self.state_size = state_size

        # Number of actions available to the agent
        self.action_size = action_size

        # Number of completed episodes
        self.number_of_episodes = 0

        # Total number of selected actions
        self.total_steps = 0

        # Replay memory stores previous experiences
        self.memory = deque(
            maxlen=MAX_MEMORY
        )

        # Use GPU when available, otherwise use CPU
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Neural network used to estimate Q-values
        self.model = LinearQNet(
            input_size=self.state_size,
            hidden_size=HIDDEN_SIZE,
            output_size=self.action_size
        ).to(self.device)

        # Object responsible for training the neural network
        self.trainer = QTrainer(
            model=self.model,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            target_update_frequency=(
                TARGET_UPDATE_FREQUENCY
            )
        )

    def get_epsilon(self):

        """
        Return the current exploration probability
        """

        # Calculate how far epsilon decay has progressed
        decay_progress = min(
            self.total_steps
            / EPSILON_DECAY_STEPS,
            1.0
        )

        # Gradually reduce epsilon
        epsilon = (
            EPSILON_START
            + decay_progress
            * (
                EPSILON_END
                - EPSILON_START
            )
        )

        return epsilon

    def get_action(
        self,
        state,
        training=True
    ):

        """
        Select one action using epsilon-greedy
        """

        # Disable exploration during evaluation
        epsilon = (
            self.get_epsilon()
            if training
            else 0.0
        )

        # Decide whether to explore
        choose_random_action = (
            training
            and random.random() < epsilon
        )

        # Select a random action for exploration
        if choose_random_action:

            action = random.randrange(
                self.action_size
            )

        # Select the action with the largest Q-value
        else:

            # Convert the state into a tensor
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            # Set the model to evaluation mode
            self.model.eval()

            # Action selection does not require gradients
            with torch.no_grad():

                # Predict one Q-value for each action
                q_values = self.model(
                    state_tensor
                )

                # Select the action with the largest Q-value
                action = int(
                    torch.argmax(
                        q_values,
                        dim=1
                    ).item()
                )

            # Return the model to training mode
            self.model.train()

        # Count actions only during training
        if training:
            self.total_steps += 1

        return action

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        """
        Store one experience in replay memory
        """

        # Convert the current state into a NumPy array
        stored_state = np.asarray(
            state,
            dtype=np.float32
        )

        # Convert the next state into a NumPy array
        stored_next_state = np.asarray(
            next_state,
            dtype=np.float32
        )

        # Create one complete RL experience
        experience = (
            stored_state,
            int(action),
            float(reward),
            stored_next_state,
            bool(done)
        )

        # Add the experience to replay memory
        self.memory.append(
            experience
        )

    def train_short_memory(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        """
        Train immediately using the newest experience
        """

        return self.trainer.train_step(

            state=np.asarray(
                state,
                dtype=np.float32
            ),

            action=int(action),

            reward=float(reward),

            next_state=np.asarray(
                next_state,
                dtype=np.float32
            ),

            done=bool(done)
        )

    def train_long_memory(self):

        """
        Train using random experiences
        from replay memory
        """

        # Do not train when replay memory is empty
        if len(self.memory) == 0:
            return None

        # Use all experiences when memory is still small
        sample_size = min(
            BATCH_SIZE,
            len(self.memory)
        )

        # Randomly select experiences from memory
        mini_sample = random.sample(
            self.memory,
            sample_size
        )

        # Separate the experience components
        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = zip(*mini_sample)

        # Combine states into one batch
        states = np.stack(
            states
        )

        # Combine next states into one batch
        next_states = np.stack(
            next_states
        )

        # Convert actions into a NumPy array
        actions = np.asarray(
            actions,
            dtype=np.int64
        )

        # Convert rewards into a NumPy array
        rewards = np.asarray(
            rewards,
            dtype=np.float32
        )

        # Convert done values into a NumPy array
        dones = np.asarray(
            dones,
            dtype=np.bool_
        )

        # Train the model using the replay batch
        return self.trainer.train_step(
            state=states,
            action=actions,
            reward=rewards,
            next_state=next_states,
            done=dones
        )

    def finish_episode(self):

        """
        Mark one episode as completed
        """

        self.number_of_episodes += 1

    def save_model(
        self,
        file_name="usv_dqn_model.pth",
        folder_path="saved_models"
    ):

        """
        Save only the neural network parameters
        """

        return self.model.save(
            file_name=file_name,
            folder_path=folder_path
        )

    def load_model(
        self,
        file_name="usv_dqn_model.pth",
        folder_path="saved_models"
    ):

        """
        Load neural network parameters
        """

        # Load the saved model parameters
        file_path = self.model.load(
            file_name=file_name,
            folder_path=folder_path,
            device=self.device
        )

        # Copy the loaded parameters into the target model
        self.trainer.update_target_model()

        # Return the model to training mode
        self.model.train()

        return file_path

    def save_checkpoint(
        self,
        file_name="training_checkpoint.pth",
        folder_path="saved_models"
    ):

        """
        Save the complete training progress
        """

        # Create the checkpoint folder if needed
        os.makedirs(
            folder_path,
            exist_ok=True
        )

        # Create the complete checkpoint path
        file_path = os.path.join(
            folder_path,
            file_name
        )

        # Store everything needed to continue training
        checkpoint = {

            # Main neural network parameters
            "model_state_dict": (
                self.model.state_dict()
            ),

            # Target neural network parameters
            "target_model_state_dict": (
                self.trainer
                .target_model
                .state_dict()
            ),

            # Adam optimizer state
            "optimizer_state_dict": (
                self.trainer
                .optimizer
                .state_dict()
            ),

            # Number of completed episodes
            "number_of_episodes": (
                self.number_of_episodes
            ),

            # Total number of selected actions
            "total_steps": (
                self.total_steps
            ),

            # Total number of model training steps
            "training_steps": (
                self.trainer.training_steps
            ),

            # State size used by the saved model
            "state_size": (
                self.state_size
            ),

            # Action size used by the saved model
            "action_size": (
                self.action_size
            )
        }

        # Save the checkpoint
        torch.save(
            checkpoint,
            file_path
        )

        return file_path

    def load_checkpoint(
        self,
        file_name="training_checkpoint.pth",
        folder_path="saved_models"
    ):

        """
        Restore the complete training progress
        """

        # Create the complete checkpoint path
        file_path = os.path.join(
            folder_path,
            file_name
        )

        # Stop when the checkpoint does not exist
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {file_path}"
            )

        # Load the checkpoint on the current device
        checkpoint = torch.load(
            file_path,
            map_location=self.device
        )

        # Read the saved state size
        saved_state_size = checkpoint.get(
            "state_size",
            self.state_size
        )

        # Read the saved action size
        saved_action_size = checkpoint.get(
            "action_size",
            self.action_size
        )

        # Check whether the environment state size changed
        if saved_state_size != self.state_size:
            raise ValueError(
                "Checkpoint state size does not "
                "match the current environment."
            )

        # Check whether the number of actions changed
        if saved_action_size != self.action_size:
            raise ValueError(
                "Checkpoint action size does not "
                "match the current environment."
            )

        # Restore the main model parameters
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Restore the target model parameters
        self.trainer.target_model.load_state_dict(
            checkpoint[
                "target_model_state_dict"
            ]
        )

        # Restore the optimizer state
        self.trainer.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        # Restore the number of completed episodes
        self.number_of_episodes = checkpoint.get(
            "number_of_episodes",
            0
        )

        # Restore the total action count
        self.total_steps = checkpoint.get(
            "total_steps",
            0
        )

        # Restore the total training step count
        self.trainer.training_steps = (
            checkpoint.get(
                "training_steps",
                0
            )
        )

        # Set the main model to training mode
        self.model.train()

        # Keep the target model in evaluation mode
        self.trainer.target_model.eval()

        return file_path