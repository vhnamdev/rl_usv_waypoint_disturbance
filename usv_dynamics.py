import math
import config

class USVDynamics:

    def clamp(self,value,minimum_value,maximum_value):

        """Limit a value inside a specified range"""

        return max(minimum_value,min(value,maximum_value))

    def normalize_angle(self,angle):

        """Keep an angle inside the range [pi, -pi]
           
           Equivalent range:
           -180 degrees to 180 degrees.
        """

        return math.atan2(math.sin(angle),math.cos(angle))

    def update(self,
        x,
        y,
        psi,
        u,
        v,
        r,
        surge_force,
        yaw_moment):

        """
            Caculate the next USV's state after one simulation step

            Current state:
                x,y : world position in meters
                psi : heading angle in radian
                u   : surge velocity in m/s
                v   : sway velocity in m/s
                r   : yaw rate in rad/s
            
            Control input:
                surge force : forward or backward force
                yaw_moment  : turning moment
            
            Return:
                new_x
                new_y
                new_psi
                new_u
                new_v
                new_r
        """

        # Limit control inputs

        limited_surge_force = self.clamp(surge_force,-config.MAX_SURGE_FORCE, config.MAX_SURGE_FORCE)
        limited_yaw_moment  = self.clamp(yaw_moment,-config.MAX_YAW_MOMENT, config.MAX_YAW_MOMENT)

        # Caculate accelerations

        surge_accelaration = (limited_surge_force - config.SURGE_DAMPING * u) / config.MASS
        sway_accelaration  = (-config.SWAY_DAMPING * v) / config.MASS
        yaw_accelaration   = (limited_yaw_moment - config.YAW_DAMPING * r) / config.YAW_INERTIA

        # Update velocities with euler method

        new_u = (u + surge_accelaration * config.DT)
        new_v = (v + sway_accelaration * config.DT)
        new_r = (r + yaw_accelaration * config.DT)

        # Update heading angle

        new_psi = (psi + new_r * config.DT)
        new_psi = self.normalize_angle(new_psi)

        # Convert body velocity to world velocity

        world_velocity_x = (new_u * math.cos(new_psi) - new_v * math.sin(new_psi))
        world_velocity_y = (new_u * math.sin(new_psi) - new_v * math.cos(new_psi))

        # Update world position

        new_x = (x + world_velocity_x * config.DT)
        new_y = (y + world_velocity_y * config.DT)

        return (new_x,new_y,new_psi,new_u,new_v,new_r)


