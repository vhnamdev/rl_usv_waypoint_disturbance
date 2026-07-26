import math
import random

import matplotlib

# Use a desktop window for Matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import config


# Turn on interactive plotting mode
plt.ion()


def generate_waypoints(
    start_x,
    start_y,
    world_width,
    world_height
):
    """
    Generate random waypoints inside the ocean.

    Every waypoint must:
    - stay away from the ocean boundary;
    - stay far enough from the starting position;
    - stay far enough from previously generated waypoints.
    """

    generated_waypoints = []

    reference_points = [
        (
            start_x,
            start_y
        )
    ]

    maximum_attempts = 10000
    attempt_count = 0

    while (
        len(generated_waypoints)
        < config.NUMBER_OF_WAYPOINTS
    ):
        attempt_count += 1

        if attempt_count > maximum_attempts:
            raise RuntimeError(
                "Unable to generate valid waypoints. "
                "Check the waypoint settings in config.py."
            )

        candidate_x = random.uniform(
            config.WAYPOINT_MARGIN,
            world_width
            - config.WAYPOINT_MARGIN
        )

        candidate_y = random.uniform(
            config.WAYPOINT_MARGIN,
            world_height
            - config.WAYPOINT_MARGIN
        )

        candidate_is_valid = True

        for reference_x, reference_y in reference_points:

            distance_to_reference = math.hypot(
                candidate_x - reference_x,
                candidate_y - reference_y
            )

            if (
                distance_to_reference
                < config.MIN_WAYPOINT_DISTANCE
            ):
                candidate_is_valid = False
                break

        if candidate_is_valid:

            new_waypoint = (
                candidate_x,
                candidate_y
            )

            generated_waypoints.append(
                new_waypoint
            )

            reference_points.append(
                new_waypoint
            )

    return generated_waypoints


def plot_training(
    episode_rewards,
    mean_rewards,
    completed_waypoints_history
):
    """
    Display live DQN training progress.

    Parameters:
        episode_rewards:
            Total reward received in each episode.

        mean_rewards:
            Mean reward calculated during training.

        completed_waypoints_history:
            Number of completed waypoints in each episode.
    """

    # Do not plot when no episode has finished
    if not episode_rewards:
        return

    # Check that all training histories have equal length
    if not (
        len(episode_rewards)
        == len(mean_rewards)
        == len(completed_waypoints_history)
    ):
        raise ValueError(
            "Training history lists must have "
            "the same length."
        )

    # Episode numbers start from 1
    episode_numbers = list(
        range(
            1,
            len(episode_rewards) + 1
        )
    )

    # Create or select the training figure
    training_figure = plt.figure(
        "USV RL Training",
        figsize=(
            10,
            7
        )
    )

    # Remove the previous plot contents
    training_figure.clear()

    # Create the reward graph
    reward_axis = training_figure.add_subplot(
        2,
        1,
        1
    )

    reward_axis.set_title(
        "USV DQN Training Progress"
    )

    reward_axis.set_xlabel(
        "Episode"
    )

    reward_axis.set_ylabel(
        "Reward"
    )

    # Draw the reward of every episode
    reward_axis.plot(
        episode_numbers,
        episode_rewards,
        label="Episode reward"
    )

    # Draw the mean reward
    reward_axis.plot(
        episode_numbers,
        mean_rewards,
        label="Mean reward"
    )

    reward_axis.grid(
        True
    )

    reward_axis.legend()

    # Display the latest episode reward
    reward_axis.text(
        episode_numbers[-1],
        episode_rewards[-1],
        f"{episode_rewards[-1]:.1f}"
    )

    # Display the latest mean reward
    reward_axis.text(
        episode_numbers[-1],
        mean_rewards[-1],
        f"{mean_rewards[-1]:.1f}"
    )

    # Create the waypoint graph
    waypoint_axis = training_figure.add_subplot(
        2,
        1,
        2
    )

    waypoint_axis.set_title(
        "Completed Waypoints"
    )

    waypoint_axis.set_xlabel(
        "Episode"
    )

    waypoint_axis.set_ylabel(
        "Waypoints"
    )

    # Keep the waypoint scale between zero
    # and the total number of waypoints
    waypoint_axis.set_ylim(
        -0.2,
        config.NUMBER_OF_WAYPOINTS + 0.5
    )

    # Use integer waypoint values on the Y axis
    waypoint_axis.set_yticks(
        range(
            0,
            config.NUMBER_OF_WAYPOINTS + 1
        )
    )

    # Draw the completed waypoint history
    waypoint_axis.plot(
        episode_numbers,
        completed_waypoints_history,
        label="Completed waypoints"
    )

    waypoint_axis.grid(
        True
    )

    waypoint_axis.legend()

    # Display the latest completed waypoint count
    waypoint_axis.text(
        episode_numbers[-1],
        completed_waypoints_history[-1],
        str(
            completed_waypoints_history[-1]
        )
    )

    # Prevent graph elements from overlapping
    training_figure.tight_layout()

    # Display the figure without stopping training
    plt.show(
        block=False
    )

    # Process Matplotlib window events
    training_figure.canvas.draw_idle()
    training_figure.canvas.flush_events()

    # Give the plot window time to update
    plt.pause(
        0.001
    )