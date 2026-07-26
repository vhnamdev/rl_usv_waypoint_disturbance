import math
import random

import config


class Disturbance:

    def __init__(self):

        # Water current speed in m/s
        self.current_speed = 0.0

        # Water current direction in radians
        self.current_direction = 0.0

        # Wind force magnitude
        self.wind_force = 0.0

        # Wind direction in radians
        self.wind_direction = 0.0

        # Turning moment caused by waves
        self.wave_moment = 0.0

        self.reset()

    def reset(self):

        """
        Create a new disturbance condition
        for the beginning of an episode
        """

        # Create a random water current
        self.current_speed = random.uniform(
            0.0,
            config.MAX_CURRENT_SPEED
        )

        self.current_direction = random.uniform(
            -math.pi,
            math.pi
        )

        # Create a random wind condition
        self.wind_force = random.uniform(
            0.0,
            config.MAX_WIND_FORCE * 0.5
        )

        self.wind_direction = random.uniform(
            -math.pi,
            math.pi
        )

        # Create a random wave moment
        self.wave_moment = random.uniform(
            -config.MAX_WAVE_MOMENT * 0.5,
            config.MAX_WAVE_MOMENT * 0.5
        )

    def update(self, psi):

        """
        Update the disturbance for one simulation step

        Input:
            psi: current USV heading angle

        Return:
            current velocity
            wind forces
            wave moment
        """

        # Slowly change the water current speed
        self.current_speed += random.uniform(
            -0.02,
            0.02
        )

        self.current_speed = max(
            0.0,
            min(
                self.current_speed,
                config.MAX_CURRENT_SPEED
            )
        )

        # Slowly change the water current direction
        self.current_direction += random.uniform(
            -0.01,
            0.01
        )

        # Keep the direction inside [-pi, pi]
        self.current_direction = math.atan2(
            math.sin(self.current_direction),
            math.cos(self.current_direction)
        )

        # Slowly change the wind force
        self.wind_force += random.uniform(
            -0.5,
            0.5
        )

        self.wind_force = max(
            0.0,
            min(
                self.wind_force,
                config.MAX_WIND_FORCE
            )
        )

        # Slowly change the wind direction
        self.wind_direction += random.uniform(
            -0.02,
            0.02
        )

        # Keep the wind direction inside [-pi, pi]
        self.wind_direction = math.atan2(
            math.sin(self.wind_direction),
            math.cos(self.wind_direction)
        )

        # Slowly change the wave moment
        self.wave_moment += random.uniform(
            -0.5,
            0.5
        )

        self.wave_moment = max(
            -config.MAX_WAVE_MOMENT,
            min(
                self.wave_moment,
                config.MAX_WAVE_MOMENT
            )
        )

        # Normally use the current wind force
        effective_wind_force = self.wind_force

        # Occasionally create a one-step wind gust
        gust_happened = (
            random.random()
            < config.GUST_PROBABILITY
        )

        if gust_happened:

            gust_force = random.uniform(
                config.MAX_WIND_FORCE * 0.5,
                config.MAX_WIND_FORCE
            )

            effective_wind_force = min(
                self.wind_force + gust_force,
                config.MAX_WIND_FORCE
            )

        # Convert water current into world X velocity
        current_velocity_x = (
            self.current_speed
            * math.cos(self.current_direction)
        )

        # Convert water current into world Y velocity
        current_velocity_y = (
            self.current_speed
            * math.sin(self.current_direction)
        )

        # Convert wind magnitude into world X force
        wind_force_x = (
            effective_wind_force
            * math.cos(self.wind_direction)
        )

        # Convert wind magnitude into world Y force
        wind_force_y = (
            effective_wind_force
            * math.sin(self.wind_direction)
        )

        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)

        # Convert world wind force into body surge force
        wind_surge_force = (
            wind_force_x * cos_psi
            + wind_force_y * sin_psi
        )

        # Convert world wind force into body sway force
        wind_sway_force = (
            -wind_force_x * sin_psi
            + wind_force_y * cos_psi
        )

        return {
            "current_velocity_x": current_velocity_x,
            "current_velocity_y": current_velocity_y,
            "wind_surge_force": wind_surge_force,
            "wind_sway_force": wind_sway_force,
            "wave_moment": self.wave_moment,
            "gust_happened": gust_happened
        }