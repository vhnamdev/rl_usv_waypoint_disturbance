import math
import config

from disturbance import Disturbance
from helper import generate_waypoints
from usv_environment import USVEnvironment
from usv_dynamics import USVDynamics


class RLEnvironment:

    def __init__(self):

        # Renderer only displays the current USV state
        self.renderer = USVEnvironment()

        # Dynamics calculates the physical motion of the USV
        self.dynamics = USVDynamics()

        # Disturbance creates water current, wind and waves
        self.disturbance = Disturbance()

        # Latest disturbance values
        self.disturbance_information = {
            "current_velocity_x": 0.0,
            "current_velocity_y": 0.0,
            "wind_surge_force": 0.0,
            "wind_sway_force": 0.0,
            "wave_moment": 0.0,
            "gust_happened": False
        }

        # Ocean dimensions converted from pixels to meters
        self.world_width = (
            config.OCEAN_WIDTH
            / config.PIXELS_PER_METER
        )

        self.world_height = (
            config.OCEAN_HEIGHT
            / config.PIXELS_PER_METER
        )

        # Initial USV position and heading
        self.start_x = self.renderer.x
        self.start_y = self.renderer.y
        self.start_psi = self.renderer.psi

        # Current USV state
        self.x = self.start_x
        self.y = self.start_y
        self.psi = self.start_psi

        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        # Waypoint information
        self.waypoints = []
        self.current_waypoint_index = 0

        # Episode information
        self.step_count = 0
        self.previous_distance = 0.0
        self.episode_done = False

        # Number of values given to the neural network
        self.state_size = 8

        # Number of discrete actions available to the RL agent
        self.action_size = 5

        self.reset()

    def get_current_waypoint(self):

        """
        Return the waypoint currently targeted by the USV

        Return None when all waypoints have been completed
        """

        if (
            self.current_waypoint_index
            >= len(self.waypoints)
        ):
            return None

        return self.waypoints[
            self.current_waypoint_index
        ]

    def get_distance_to_waypoint(self):

        """
        Calculate the distance from the USV
        to the current waypoint
        """

        current_waypoint = (
            self.get_current_waypoint()
        )

        if current_waypoint is None:
            return 0.0

        waypoint_x, waypoint_y = (
            current_waypoint
        )

        difference_x = waypoint_x - self.x
        difference_y = waypoint_y - self.y

        distance = math.hypot(
            difference_x,
            difference_y
        )

        return distance

    def get_heading_error(self):

        """
        Calculate the shortest angle from the current
        USV heading to the current waypoint direction
        """

        current_waypoint = (
            self.get_current_waypoint()
        )

        if current_waypoint is None:
            return 0.0

        waypoint_x, waypoint_y = (
            current_waypoint
        )

        difference_x = waypoint_x - self.x
        difference_y = waypoint_y - self.y

        # Direction from the USV to the waypoint
        desired_heading = math.atan2(
            difference_y,
            difference_x
        )

        # Difference between desired and current heading
        heading_error = (
            desired_heading
            - self.psi
        )

        # Keep the heading error inside [-pi, pi]
        return self.dynamics.normalize_angle(
            heading_error
        )

    def get_state(self):

        """
        Return the state observed by the RL agent

        State:
            0. Normalized difference in X
            1. Normalized difference in Y
            2. Normalized distance to the waypoint
            3. Sine of heading error
            4. Cosine of heading error
            5. Normalized surge velocity
            6. Normalized sway velocity
            7. Normalized yaw rate
        """

        current_waypoint = (
            self.get_current_waypoint()
        )

        # Use a neutral state after all waypoints are completed
        if current_waypoint is None:

            normalized_difference_x = 0.0
            normalized_difference_y = 0.0
            normalized_distance = 0.0

            heading_error_sine = 0.0
            heading_error_cosine = 1.0

        else:

            waypoint_x, waypoint_y = (
                current_waypoint
            )

            difference_x = waypoint_x - self.x
            difference_y = waypoint_y - self.y

            distance = math.hypot(
                difference_x,
                difference_y
            )

            maximum_possible_distance = math.hypot(
                self.world_width,
                self.world_height
            )

            heading_error = (
                self.get_heading_error()
            )

            # Normalize waypoint position information
            normalized_difference_x = (
                difference_x
                / self.world_width
            )

            normalized_difference_y = (
                difference_y
                / self.world_height
            )

            normalized_distance = (
                distance
                / maximum_possible_distance
            )

            # Represent the angle without the -pi and pi discontinuity
            heading_error_sine = math.sin(
                heading_error
            )

            heading_error_cosine = math.cos(
                heading_error
            )

        # Normalize USV velocities
        normalized_u = (
            self.u
            / config.MAX_SURGE_SPEED
        )

        normalized_v = (
            self.v
            / config.MAX_SWAY_SPEED
        )

        normalized_r = (
            self.r
            / config.MAX_YAW_RATE
        )

        state = [
            normalized_difference_x,
            normalized_difference_y,
            normalized_distance,
            heading_error_sine,
            heading_error_cosine,
            normalized_u,
            normalized_v,
            normalized_r
        ]

        return state

    def decode_action(self, action):

        """
        Convert a discrete RL action into:

            surge_force
            yaw_moment

        Actions:
            0: Coast
            1: Move forward slowly
            2: Move forward quickly
            3: Move forward and turn left
            4: Move forward and turn right
        """

        # Convert a PyTorch tensor into Python data
        if hasattr(action, "detach"):
            action = (
                action
                .detach()
                .cpu()
                .tolist()
            )

        # Convert a NumPy array into Python data
        elif hasattr(action, "tolist"):
            action = action.tolist()

        # Remove an extra batch dimension
        #
        # Example:
        # [[0, 1, 0, 0, 0]]
        #
        # Becomes:
        # [0, 1, 0, 0, 0]
        if (
            isinstance(action, (list, tuple))
            and len(action) == 1
            and isinstance(
                action[0],
                (list, tuple)
            )
        ):
            action = action[0]

        # Convert a one-hot action into an action index
        if isinstance(action, (list, tuple)):

            if len(action) == self.action_size:

                action = max(
                    range(self.action_size),
                    key=lambda index: action[index]
                )

            elif len(action) == 1:
                action = action[0]

            else:
                raise ValueError(
                    f"Invalid action format: {action}"
                )

        action = int(action)

        action_map = {

            # Action 0: Do not produce force or moment
            0: (
                0.0,
                0.0
            ),

            # Action 1: Move forward slowly
            1: (
                config.MAX_SURGE_FORCE * 0.50,
                0.0
            ),

            # Action 2: Move forward at maximum force
            2: (
                config.MAX_SURGE_FORCE,
                0.0
            ),

            # Action 3: Move forward and turn left
            3: (
                config.MAX_SURGE_FORCE * 0.75,
                -config.MAX_YAW_MOMENT
            ),

            # Action 4: Move forward and turn right
            4: (
                config.MAX_SURGE_FORCE * 0.75,
                config.MAX_YAW_MOMENT
            )
        }

        if action not in action_map:
            raise ValueError(
                f"Invalid action index: {action}"
            )

        surge_force, yaw_moment = (
            action_map[action]
        )

        return surge_force, yaw_moment

    def is_out_of_bounds(self):

        """
        Check whether the center of the USV
        has left the ocean area
        """

        is_inside_ocean = (
            0.0 <= self.x <= self.world_width
            and
            0.0 <= self.y <= self.world_height
        )

        return not is_inside_ocean

    def calculate_reward(
        self,
        current_distance,
        surge_force,
        yaw_moment
    ):

        """
        Calculate the reward after one action
        """

        # Positive when the USV moves closer
        # to the current waypoint
        distance_progress = (
            self.previous_distance
            - current_distance
        )

        progress_reward = (
            distance_progress * 20.0
        )

        # Positive when the USV points toward the waypoint
        heading_error = (
            self.get_heading_error()
        )

        heading_reward = (
            math.cos(heading_error)
            * 0.01
        )

        # Prevent the agent from standing still forever
        time_penalty = -0.02

        # Normalize the control input magnitude
        normalized_force = (
            abs(surge_force)
            / config.MAX_SURGE_FORCE
        )

        normalized_moment = (
            abs(yaw_moment)
            / config.MAX_YAW_MOMENT
        )

        # Penalize unnecessary large control commands
        control_penalty = -0.002 * (
            normalized_force
            + normalized_moment
        )

        reward = (
            progress_reward
            + heading_reward
            + time_penalty
            + control_penalty
        )

        return reward

    def synchronize_renderer(self):

        """
        Copy the latest USV state and waypoint data
        to the renderer
        """

        # Copy the current USV position and heading
        self.renderer.x = self.x
        self.renderer.y = self.y
        self.renderer.psi = self.psi

        # Copy the current USV velocities
        self.renderer.u = self.u
        self.renderer.v = self.v
        self.renderer.r = self.r

        # Copy the waypoint list to the renderer
        self.renderer.waypoints = list(
            self.waypoints
        )

        # Copy the current waypoint index to the renderer
        self.renderer.current_waypoint_index = (
            self.current_waypoint_index
        )

    def reset(self):

        """
        Reset the environment and start a new episode

        Return the first state of the new episode
        """

        # Reset position and heading
        self.x = self.start_x
        self.y = self.start_y
        self.psi = self.start_psi

        # Reset velocities
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        # Create a new disturbance condition
        self.disturbance.reset()

        # Clear the previous disturbance values
        self.disturbance_information = {
            "current_velocity_x": 0.0,
            "current_velocity_y": 0.0,
            "wind_surge_force": 0.0,
            "wind_sway_force": 0.0,
            "wave_moment": 0.0,
            "gust_happened": False
        }

        # Generate a new waypoint set
        self.waypoints = generate_waypoints(
            self.start_x,
            self.start_y,
            self.world_width,
            self.world_height
        )

        self.current_waypoint_index = 0

        # Reset episode information
        self.step_count = 0
        self.episode_done = False

        # Store the initial distance
        self.previous_distance = (
            self.get_distance_to_waypoint()
        )

        # Send the reset state and waypoints to the renderer
        self.synchronize_renderer()

        initial_state = self.get_state()

        return initial_state

    def step(self, action):

        """
        Execute one RL action

        Return:
            next_state
            reward
            done
            information
        """

        if self.episode_done:
            raise RuntimeError(
                "The episode has already finished. "
                "Call reset() before calling step() again."
            )

        # Convert the selected action into physical control inputs
        surge_force, yaw_moment = (
            self.decode_action(action)
        )

        # Update current, wind and wave disturbances
        self.disturbance_information = (
            self.disturbance.update(
                self.psi
            )
        )

        # Calculate the next physical state of the USV
        (
            self.x,
            self.y,
            self.psi,
            self.u,
            self.v,
            self.r
        ) = self.dynamics.update(
            self.x,
            self.y,
            self.psi,
            self.u,
            self.v,
            self.r,
            surge_force,
            yaw_moment,

            current_velocity_x=(
                self.disturbance_information[
                    "current_velocity_x"
                ]
            ),

            current_velocity_y=(
                self.disturbance_information[
                    "current_velocity_y"
                ]
            ),

            wind_surge_force=(
                self.disturbance_information[
                    "wind_surge_force"
                ]
            ),

            wind_sway_force=(
                self.disturbance_information[
                    "wind_sway_force"
                ]
            ),

            wave_moment=(
                self.disturbance_information[
                    "wave_moment"
                ]
            )
        )

        # Count one completed environment step
        self.step_count += 1

        # Measure the distance after performing the action
        distance_to_target = (
            self.get_distance_to_waypoint()
        )

        # Calculate the normal reward for this step
        reward = self.calculate_reward(
            distance_to_target,
            surge_force,
            yaw_moment
        )

        waypoint_reached = False
        all_waypoints_completed = False
        timed_out = False

        out_of_bounds = (
            self.is_out_of_bounds()
        )

        # End the episode when the USV leaves the ocean
        if out_of_bounds:

            reward -= 200.0
            self.episode_done = True

        # Check whether the current waypoint was reached
        elif (
            distance_to_target
            <= config.WAYPOINT_RADIUS
        ):

            waypoint_reached = True

            reward += 100.0

            self.current_waypoint_index += 1

            # All waypoints have been completed
            if (
                self.current_waypoint_index
                >= len(self.waypoints)
            ):

                all_waypoints_completed = True

                reward += 300.0

                self.previous_distance = 0.0
                self.episode_done = True

            # Continue toward the next waypoint
            else:

                self.previous_distance = (
                    self.get_distance_to_waypoint()
                )

        # The current waypoint has not been reached
        else:

            self.previous_distance = (
                distance_to_target
            )

        # End the episode when it takes too many steps
        if (
            self.step_count >= config.MAX_STEPS
            and not self.episode_done
        ):

            timed_out = True

            reward -= 100.0
            self.episode_done = True

        # Copy the new state and waypoint index to the renderer
        self.synchronize_renderer()

        next_state = self.get_state()

        information = {

            "waypoint_index": (
                self.current_waypoint_index
            ),

            "number_of_waypoints": len(
                self.waypoints
            ),

            "distance_to_waypoint": (
                self.get_distance_to_waypoint()
            ),

            "heading_error": (
                self.get_heading_error()
            ),

            "waypoint_reached": (
                waypoint_reached
            ),

            "all_waypoints_completed": (
                all_waypoints_completed
            ),

            "out_of_bounds": (
                out_of_bounds
            ),

            "timed_out": (
                timed_out
            ),

            "surge_force": (
                surge_force
            ),

            "yaw_moment": (
                yaw_moment
            ),

            "current_velocity_x": (
                self.disturbance_information[
                    "current_velocity_x"
                ]
            ),

            "current_velocity_y": (
                self.disturbance_information[
                    "current_velocity_y"
                ]
            ),

            "wind_surge_force": (
                self.disturbance_information[
                    "wind_surge_force"
                ]
            ),

            "wind_sway_force": (
                self.disturbance_information[
                    "wind_sway_force"
                ]
            ),

            "wave_moment": (
                self.disturbance_information[
                    "wave_moment"
                ]
            ),

            "gust_happened": (
                self.disturbance_information[
                    "gust_happened"
                ]
            ),

            "step_count": (
                self.step_count
            )
        }

        return (
            next_state,
            reward,
            self.episode_done,
            information
        )