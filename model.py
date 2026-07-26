import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim


class LinearQNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Neural network used to estimate Q-values
        self.network = nn.Sequential(

            # Convert the state into hidden features
            nn.Linear(
                input_size,
                hidden_size
            ),

            # Add non-linearity
            nn.ReLU(),

            # Reduce the hidden feature size
            nn.Linear(
                hidden_size,
                hidden_size // 2
            ),

            # Add non-linearity
            nn.ReLU(),

            # Produce one Q-value for each action
            nn.Linear(
                hidden_size // 2,
                output_size
            )
        )

    def forward(self, state):

        """
        Convert a state into Q-values
        """

        return self.network(state)

    def save(self, file_name="usv_dqn_model.pth", folder_path="saved_models"):

        """
        Save the neural network parameters
        """

        # Create the model folder if it does not exist
        os.makedirs(
            folder_path,
            exist_ok=True
        )

        # Create the complete model file path
        file_path = os.path.join(
            folder_path,
            file_name
        )

        # Save only the neural network parameters
        torch.save(
            self.state_dict(),
            file_path
        )

        return file_path

    def load( self, file_name="usv_dqn_model.pth", folder_path="saved_models", device=None):

        """
        Load the neural network parameters
        """

        # Create the complete model file path
        file_path = os.path.join(
            folder_path,
            file_name
        )

        # Stop when the model file does not exist
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found: {file_path}"
            )

        # Use the current model device by default
        if device is None:
            device = next(
                self.parameters()
            ).device

        # Load the saved parameters
        model_state = torch.load(
            file_path,
            map_location=device
        )

        # Copy the saved parameters into the model
        self.load_state_dict(
            model_state
        )

        # Set the model to evaluation mode
        self.eval()

        return file_path


class QTrainer:

    def __init__(
        self,
        model,
        learning_rate,
        gamma,
        target_update_frequency=1000
    ):

        # Main neural network being trained
        self.model = model

        # Discount factor used in the Bellman equation
        self.gamma = gamma

        # Number of training steps between target model updates
        self.target_update_frequency = (
            target_update_frequency
        )

        # Number of completed training steps
        self.training_steps = 0

        # Device currently used by the model
        self.device = next(
            self.model.parameters()
        ).device

        # Stable copy of the main model
        self.target_model = copy.deepcopy(
            self.model
        ).to(self.device)

        # The target model is only used for prediction
        self.target_model.eval()

        # Adam optimizer updates the model parameters
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Huber loss is less sensitive to large errors
        self.criterion = nn.SmoothL1Loss()

    def update_target_model(self):

        """
        Copy the main model parameters
        into the target model
        """

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    def train_step(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        """
        Perform one DQN training step
        """

        # Convert the current states into a tensor
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        )

        # Convert the next states into a tensor
        next_state_tensor = torch.as_tensor(
            next_state,
            dtype=torch.float32,
            device=self.device
        )

        # Convert the selected actions into a tensor
        action_tensor = torch.as_tensor(
            action,
            dtype=torch.long,
            device=self.device
        ).view(-1)

        # Convert the rewards into a tensor
        reward_tensor = torch.as_tensor(
            reward,
            dtype=torch.float32,
            device=self.device
        ).view(-1)

        # Convert the episode results into a tensor
        done_tensor = torch.as_tensor(
            done,
            dtype=torch.bool,
            device=self.device
        ).view(-1)

        # Add a batch dimension for one state
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Add a batch dimension for one next state
        if next_state_tensor.ndim == 1:
            next_state_tensor = (
                next_state_tensor.unsqueeze(0)
            )

        # Number of experiences in the current batch
        batch_size = state_tensor.shape[0]

        # Check the action batch size
        if action_tensor.shape[0] != batch_size:
            raise ValueError(
                "Action batch size does not match "
                "state batch size."
            )

        # Check the reward batch size
        if reward_tensor.shape[0] != batch_size:
            raise ValueError(
                "Reward batch size does not match "
                "state batch size."
            )

        # Check the done batch size
        if done_tensor.shape[0] != batch_size:
            raise ValueError(
                "Done batch size does not match "
                "state batch size."
            )

        # Predict Q-values for the current states
        prediction = self.model(
            state_tensor
        )

        # Create a target from the current prediction
        target = prediction.detach().clone()

        # Target calculations do not need gradients
        with torch.no_grad():

            # Predict Q-values for the next states
            next_q_values = self.target_model(
                next_state_tensor
            )

            # Select the largest next Q-value
            maximum_next_q_value = (
                next_q_values
                .max(dim=1)
                .values
            )

            # Use zero future reward when the episode is done
            not_done = (
                ~done_tensor
            ).float()

            # Calculate the Bellman target
            target_q_value = (
                reward_tensor
                + self.gamma
                * maximum_next_q_value
                * not_done
            )

        # Create one index for each experience
        batch_indices = torch.arange(
            batch_size,
            device=self.device
        )

        # Update only the Q-value of the selected action
        target[
            batch_indices,
            action_tensor
        ] = target_q_value

        # Remove gradients from the previous training step
        self.optimizer.zero_grad()

        # Compare predicted Q-values with target Q-values
        loss = self.criterion(
            prediction,
            target
        )

        # Calculate gradients
        loss.backward()

        # Prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=10.0
        )

        # Update the neural network parameters
        self.optimizer.step()

        # Count the completed training step
        self.training_steps += 1

        # Periodically update the target model
        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):
            self.update_target_model()

        return loss.item()