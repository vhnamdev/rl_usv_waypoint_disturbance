import os

from agent import Agent
from rl_environment import RLEnvironment


# Number of evaluation episodes
NUMBER_OF_EVALUATION_EPISODES = 10

# Folder containing the saved models
CHECKPOINT_FOLDER = "checkpoints"

# Best model saved during training
BEST_MODEL_FILE = "best_model.pth"

# Complete training checkpoint used as a fallback
TRAINING_CHECKPOINT_FILE = "training_checkpoint.pth"


def evaluate():

    # Create the RL environment
    environment = RLEnvironment()

    # Create the agent with the correct input and output sizes
    agent = Agent(
        state_size=environment.state_size,
        action_size=environment.action_size
    )

    # Create the saved file paths
    best_model_path = os.path.join(
        CHECKPOINT_FOLDER,
        BEST_MODEL_FILE
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_FOLDER,
        TRAINING_CHECKPOINT_FILE
    )

    # Load the best model when available
    if os.path.exists(best_model_path):

        agent.load_model(
            file_name=BEST_MODEL_FILE,
            folder_path=CHECKPOINT_FOLDER
        )

        print(
            f"Loaded model: {best_model_path}"
        )

    # Use the latest checkpoint when no best model exists
    elif os.path.exists(checkpoint_path):

        agent.load_checkpoint(
            file_name=TRAINING_CHECKPOINT_FILE,
            folder_path=CHECKPOINT_FOLDER
        )

        print(
            f"Loaded checkpoint: {checkpoint_path}"
        )

    else:

        environment.renderer.close()

        raise FileNotFoundError(
            "No trained model was found. "
            "Run train.py before evaluate.py."
        )

    # The model is not being trained during evaluation
    agent.model.eval()

    try:

        for episode_index in range(
            NUMBER_OF_EVALUATION_EPISODES
        ):

            if not environment.renderer.running:
                break

            # Start a new evaluation episode
            state = environment.reset()

            done = False
            episode_reward = 0.0

            information = {}

            while (
                not done
                and environment.renderer.running
            ):

                # Process the window close event
                environment.renderer.handle_events()

                if not environment.renderer.running:
                    break

                # Select only the action with the largest Q-value
                action = agent.get_action(
                    state,
                    training=False
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

                # Move to the next state
                state = next_state

                # Record the total reward
                episode_reward += reward

                # Display the current state
                environment.renderer.render()

            if not environment.renderer.running:
                break

            waypoints_reached = information.get(
                "waypoint_index",
                0
            )

            number_of_waypoints = information.get(
                "number_of_waypoints",
                0
            )

            if information.get(
                "all_waypoints_completed",
                False
            ):
                waypoints_reached = (
                    number_of_waypoints
                )

            print(
                f"Evaluation episode: "
                f"{episode_index + 1} | "
                f"Reward: {episode_reward:.2f} | "
                f"Waypoints: "
                f"{waypoints_reached}/"
                f"{number_of_waypoints} | "
                f"Steps: "
                f"{information.get('step_count', 0)} | "
                f"Completed: "
                f"{information.get(
                    'all_waypoints_completed',
                    False
                )}"
            )

    except KeyboardInterrupt:

        print(
            "\nEvaluation stopped by the user."
        )

    finally:

        # Close the Pygame window
        environment.renderer.close()


if __name__ == "__main__":
    evaluate()