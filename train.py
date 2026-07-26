import os

from agent import Agent
from rl_environment import RLEnvironment
from helper import plot_training


# Total number of training episodes
NUMBER_OF_EPISODES = 1000

# Display every training episode
RENDER_EVERY_EPISODES = 1

# Save the training checkpoint after this interval
CHECKPOINT_INTERVAL = 25

# Continue training from the saved checkpoint when possible
CONTINUE_TRAINING = True

# Folder used to save models and checkpoints
CHECKPOINT_FOLDER = "checkpoints"

# File containing the best neural network parameters
BEST_MODEL_FILE = "best_model.pth"

# File containing the complete training progress
TRAINING_CHECKPOINT_FILE = "training_checkpoint.pth"


def train():

    # Create the RL environment
    environment = RLEnvironment()

    # Create the RL agent using the environment sizes
    agent = Agent(
        state_size=environment.state_size,
        action_size=environment.action_size
    )

    # Create the complete checkpoint path
    checkpoint_path = os.path.join(
        CHECKPOINT_FOLDER,
        TRAINING_CHECKPOINT_FILE
    )

    # Best training results recorded so far
    best_waypoints_reached = -1
    best_episode_reward = float("-inf")

    # Total reward of every completed episode
    episode_rewards = []

    # Mean reward history
    mean_rewards = []

    # Number of completed waypoints in every episode
    completed_waypoints_history = []

    # Continue from the previous checkpoint when available
    if (
        CONTINUE_TRAINING
        and os.path.exists(checkpoint_path)
    ):
        agent.load_checkpoint(
            file_name=TRAINING_CHECKPOINT_FILE,
            folder_path=CHECKPOINT_FOLDER
        )

        print(
            "Training checkpoint loaded."
        )

        print(
            f"Continue from episode "
            f"{agent.number_of_episodes + 1}."
        )

    try:

        # Train until the requested number of episodes is reached
        while (
            agent.number_of_episodes
            < NUMBER_OF_EPISODES
            and environment.renderer.running
        ):

            # Start a new episode
            state = environment.reset()

            done = False

            episode_reward = 0.0
            episode_losses = []

            information = {}

            # Decide whether this episode should be displayed
            render_episode = (
                agent.number_of_episodes == 0
                or
                (
                    agent.number_of_episodes + 1
                )
                % RENDER_EVERY_EPISODES
                == 0
            )

            # Display the reset state and new waypoints
            if render_episode:
                environment.renderer.render()

            while (
                not done
                and environment.renderer.running
            ):

                # Process the Pygame close event
                environment.renderer.handle_events()

                if not environment.renderer.running:
                    break

                # Select one action using epsilon-greedy
                action = agent.get_action(
                    state,
                    training=True
                )

                # Execute the selected action
                (
                    next_state,
                    reward,
                    done,
                    information
                ) = environment.step(
                    action
                )

                # Train immediately using the newest experience
                short_memory_loss = (
                    agent.train_short_memory(
                        state,
                        action,
                        reward,
                        next_state,
                        done
                    )
                )

                # Store the experience in replay memory
                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                # Move to the next state
                state = next_state

                # Add the reward to the current episode total
                episode_reward += reward

                # Record the training loss
                if short_memory_loss is not None:
                    episode_losses.append(
                        short_memory_loss
                    )

                # Display the current episode
                if render_episode:
                    environment.renderer.render()

            # Stop training when the window is closed
            if not environment.renderer.running:
                break

            # Train using random experiences from replay memory
            long_memory_loss = (
                agent.train_long_memory()
            )

            # Mark the episode as completed
            agent.finish_episode()

            # Read the number of reached waypoints
            waypoints_reached = information.get(
                "waypoint_index",
                0
            )

            number_of_waypoints = information.get(
                "number_of_waypoints",
                0
            )

            # All waypoints were reached
            if information.get(
                "all_waypoints_completed",
                False
            ):
                waypoints_reached = (
                    number_of_waypoints
                )

            # Calculate the average short-memory loss
            if episode_losses:
                average_short_loss = (
                    sum(episode_losses)
                    / len(episode_losses)
                )
            else:
                average_short_loss = 0.0

            # Replace a missing long-memory loss
            if long_memory_loss is None:
                long_memory_loss = 0.0

            # Save this episode reward
            episode_rewards.append(
                episode_reward
            )

            # Save the number of completed waypoints
            completed_waypoints_history.append(
                waypoints_reached
            )

            # Get at most the latest 100 episode rewards
            recent_rewards = (
                episode_rewards[-100:]
            )

            # Calculate the mean reward
            mean_reward = (
                sum(recent_rewards)
                / len(recent_rewards)
            )

            # Save the current mean reward
            mean_rewards.append(
                mean_reward
            )

            # Update the live Matplotlib training graph
            plot_training(
                episode_rewards,
                mean_rewards,
                completed_waypoints_history
            )

            # Check whether this episode is better
            better_result = (
                waypoints_reached
                > best_waypoints_reached
                or
                (
                    waypoints_reached
                    == best_waypoints_reached
                    and episode_reward
                    > best_episode_reward
                )
            )

            # Save the best model
            if better_result:

                best_waypoints_reached = (
                    waypoints_reached
                )

                best_episode_reward = (
                    episode_reward
                )

                agent.save_model(
                    file_name=BEST_MODEL_FILE,
                    folder_path=CHECKPOINT_FOLDER
                )

                print(
                    "Best model saved."
                )

            # Save the complete training progress periodically
            if (
                agent.number_of_episodes
                % CHECKPOINT_INTERVAL
                == 0
            ):
                agent.save_checkpoint(
                    file_name=TRAINING_CHECKPOINT_FILE,
                    folder_path=CHECKPOINT_FOLDER
                )

                print(
                    "Training checkpoint saved."
                )

            # Display the current training result
            print(
                f"Episode: "
                f"{agent.number_of_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Mean reward: {mean_reward:.2f} | "
                f"Waypoints: "
                f"{waypoints_reached}/"
                f"{number_of_waypoints} | "
                f"Steps: "
                f"{information.get('step_count', 0)} | "
                f"Epsilon: "
                f"{agent.get_epsilon():.4f} | "
                f"Short loss: "
                f"{average_short_loss:.6f} | "
                f"Long loss: "
                f"{long_memory_loss:.6f}"
            )

    except KeyboardInterrupt:

        print(
            "\nTraining stopped by the user."
        )

    finally:

        # Save the latest training progress before closing
        agent.save_checkpoint(
            file_name=TRAINING_CHECKPOINT_FILE,
            folder_path=CHECKPOINT_FOLDER
        )

        # Close the Pygame window
        environment.renderer.close()

        print(
            "Latest training checkpoint saved."
        )


if __name__ == "__main__":
    train()