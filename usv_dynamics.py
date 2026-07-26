import math
import config


class USVDynamics:

    def clamp(
        self,
        value,
        minimum_value,
        maximum_value
    ):

        """
        Limit a value inside a specified range
        """

        return max(
            minimum_value,
            min(value, maximum_value)
        )

    def normalize_angle(self, angle):

        """
        Keep an angle inside the range [-pi, pi]

        Equivalent range:
        -180 degrees to 180 degrees
        """

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    def update(
        self,
        x,
        y,
        psi,
        u,
        v,
        r,
        surge_force,
        yaw_moment,
        current_velocity_x=0.0,
        current_velocity_y=0.0,
        wind_surge_force=0.0,
        wind_sway_force=0.0,
        wave_moment=0.0
    ):

        """
        Calculate the next USV state
        after one simulation step
        """

        # Limit the control force produced by the RL agent
        limited_surge_force = self.clamp(
            surge_force,
            -config.MAX_SURGE_FORCE,
            config.MAX_SURGE_FORCE
        )

        # Limit the control moment produced by the RL agent
        limited_yaw_moment = self.clamp(
            yaw_moment,
            -config.MAX_YAW_MOMENT,
            config.MAX_YAW_MOMENT
        )

        # Combine the RL force with the wind force
        total_surge_force = (
            limited_surge_force
            + wind_surge_force
        )

        # The current model has no sway actuator
        # Wind is the external sway force
        total_sway_force = (
            wind_sway_force
        )

        # Combine the RL moment with the wave moment
        total_yaw_moment = (
            limited_yaw_moment
            + wave_moment
        )

        # Calculate surge acceleration
        surge_acceleration = (
            total_surge_force
            - config.SURGE_DAMPING * u
        ) / config.MASS

        # Calculate sway acceleration
        sway_acceleration = (
            total_sway_force
            - config.SWAY_DAMPING * v
        ) / config.MASS

        # Calculate yaw acceleration
        yaw_acceleration = (
            total_yaw_moment
            - config.YAW_DAMPING * r
        ) / config.YAW_INERTIA

        # Update surge velocity using Euler integration
        new_u = (
            u
            + surge_acceleration * config.DT
        )

        # Update sway velocity using Euler integration
        new_v = (
            v
            + sway_acceleration * config.DT
        )

        # Update yaw rate using Euler integration
        new_r = (
            r
            + yaw_acceleration * config.DT
        )

        # Limit surge velocity
        new_u = self.clamp(
            new_u,
            -config.MAX_SURGE_SPEED,
            config.MAX_SURGE_SPEED
        )

        # Limit sway velocity
        new_v = self.clamp(
            new_v,
            -config.MAX_SWAY_SPEED,
            config.MAX_SWAY_SPEED
        )

        # Limit yaw rate
        new_r = self.clamp(
            new_r,
            -config.MAX_YAW_RATE,
            config.MAX_YAW_RATE
        )

        # Update the heading angle
        new_psi = (
            psi
            + new_r * config.DT
        )

        new_psi = self.normalize_angle(
            new_psi
        )

        # Convert body velocity into world X velocity
        world_velocity_x = (
            new_u * math.cos(new_psi)
            - new_v * math.sin(new_psi)
        )

        # Convert body velocity into world Y velocity
        world_velocity_y = (
            new_u * math.sin(new_psi)
            + new_v * math.cos(new_psi)
        )

        # Add the water current to world velocity
        world_velocity_x += current_velocity_x
        world_velocity_y += current_velocity_y

        # Update world X position
        new_x = (
            x
            + world_velocity_x * config.DT
        )

        # Update world Y position
        new_y = (
            y
            + world_velocity_y * config.DT
        )

        return (
            new_x,
            new_y,
            new_psi,
            new_u,
            new_v,
            new_r
        )