import math
import pygame
import config


class USVEnvironment:

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                config.SCREEN_WIDTH,
                config.SCREEN_HEIGHT
            )
        )

        pygame.display.set_caption(
            "USV Multi-Waypoint RL Control"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "arial",
            22
        )

        self.small_font = pygame.font.SysFont(
            "arial",
            17
        )

        # USV position in meters
        self.x = 40.0
        self.y = 40.0

        # Heading angle in radians
        self.psi = 0.0

        # Velocities
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        # Waypoint data received from RLEnvironment
        self.waypoints = []

        # Number of completed waypoints
        self.current_waypoint_index = 0

        self.running = True

    def handle_events(self):

        # Read all Pygame events
        for event in pygame.event.get():

            # Stop the program when the window is closed
            if event.type == pygame.QUIT:
                self.running = False

    def world_to_screen(
        self,
        x_in_meters,
        y_in_meters
    ):

        # Convert the world X position from meters to pixels
        screen_x = int(
            x_in_meters
            * config.PIXELS_PER_METER
        )

        # Convert the world Y position from meters to pixels
        screen_y = int(
            y_in_meters
            * config.PIXELS_PER_METER
        )

        return screen_x, screen_y

    def draw_ocean(self):

        # Create the ocean drawing area
        ocean_rect = pygame.Rect(
            0,
            0,
            config.OCEAN_WIDTH,
            config.OCEAN_HEIGHT
        )

        # Draw the ocean background
        pygame.draw.rect(
            self.screen,
            config.OCEAN_COLOR,
            ocean_rect
        )

    def draw_grid(self):

        # Distance between two grid lines in meters
        grid_spacing_meters = 5.0

        # Convert grid spacing from meters to pixels
        grid_spacing_pixels = int(
            grid_spacing_meters
            * config.PIXELS_PER_METER
        )

        # Draw vertical grid lines
        for x_pixel in range(
            0,
            config.OCEAN_WIDTH,
            grid_spacing_pixels
        ):
            pygame.draw.line(
                self.screen,
                config.GRID_COLOR,
                (
                    x_pixel,
                    0
                ),
                (
                    x_pixel,
                    config.OCEAN_HEIGHT
                ),
                1
            )

        # Draw horizontal grid lines
        for y_pixel in range(
            0,
            config.OCEAN_HEIGHT,
            grid_spacing_pixels
        ):
            pygame.draw.line(
                self.screen,
                config.GRID_COLOR,
                (
                    0,
                    y_pixel
                ),
                (
                    config.OCEAN_WIDTH,
                    y_pixel
                ),
                1
            )

    def draw_waypoints(self):

        # Do not draw when no waypoint data is available
        if not self.waypoints:
            return

        # Convert waypoint radius from meters to pixels
        waypoint_radius_pixels = max(
            8,
            int(
                config.WAYPOINT_RADIUS
                * config.PIXELS_PER_METER
            )
        )

        # Convert all waypoint positions to screen positions
        waypoint_screen_points = []

        for waypoint_x, waypoint_y in self.waypoints:

            waypoint_screen_point = (
                self.world_to_screen(
                    waypoint_x,
                    waypoint_y
                )
            )

            waypoint_screen_points.append(
                waypoint_screen_point
            )

        # Draw the reference path between the waypoints
        if len(waypoint_screen_points) >= 2:

            pygame.draw.lines(
                self.screen,
                config.REFERENCE_PATH_COLOR,
                False,
                waypoint_screen_points,
                2
            )

        # Draw a line from the USV to the current waypoint
        if (
            self.current_waypoint_index
            < len(self.waypoints)
        ):

            usv_screen_position = (
                self.world_to_screen(
                    self.x,
                    self.y
                )
            )

            current_waypoint_position = (
                waypoint_screen_points[
                    self.current_waypoint_index
                ]
            )

            pygame.draw.line(
                self.screen,
                config.CURRENT_WAYPOINT_COLOR,
                usv_screen_position,
                current_waypoint_position,
                2
            )

        # Draw every waypoint
        for waypoint_index, screen_position in enumerate(
            waypoint_screen_points
        ):

            # Waypoint already completed
            if (
                waypoint_index
                < self.current_waypoint_index
            ):
                waypoint_color = (
                    config.COMPLETED_WAYPOINT_COLOR
                )

            # Current target waypoint
            elif (
                waypoint_index
                == self.current_waypoint_index
            ):
                waypoint_color = (
                    config.CURRENT_WAYPOINT_COLOR
                )

            # Future waypoint
            else:
                waypoint_color = (
                    config.FUTURE_WAYPOINT_COLOR
                )

            # Draw the waypoint circle
            pygame.draw.circle(
                self.screen,
                waypoint_color,
                screen_position,
                waypoint_radius_pixels,
                3
            )

            # Create the waypoint number
            waypoint_number_surface = (
                self.small_font.render(
                    str(waypoint_index + 1),
                    True,
                    config.TEXT_COLOR
                )
            )

            # Place the number at the center of the waypoint
            waypoint_number_rect = (
                waypoint_number_surface.get_rect(
                    center=screen_position
                )
            )

            # Draw the waypoint number
            self.screen.blit(
                waypoint_number_surface,
                waypoint_number_rect
            )

    def draw_panel(self):

        # Create the right-side panel area
        panel_rect = pygame.Rect(
            config.OCEAN_WIDTH,
            0,
            config.PANEL_WIDTH,
            config.SCREEN_HEIGHT
        )

        # Draw the panel background
        pygame.draw.rect(
            self.screen,
            config.PANEL_COLOR,
            panel_rect
        )

        panel_x = config.OCEAN_WIDTH + 25
        panel_y = 35
        line_spacing = 34

        # Total number of waypoints
        number_of_waypoints = len(
            self.waypoints
        )

        # Number of completed waypoints
        completed_waypoints = min(
            self.current_waypoint_index,
            number_of_waypoints
        )

        # Create the waypoint progress text
        if not self.waypoints:

            waypoint_information = "0/0"

        else:

            waypoint_information = (
                f"{completed_waypoints}"
                f"/{number_of_waypoints}"
            )

        information_lines = [
            "USV INFORMATION",
            "",
            f"Position X: {self.x:.2f} m",
            f"Position Y: {self.y:.2f} m",
            (
                f"Heading: "
                f"{math.degrees(self.psi):.2f} deg"
            ),
            f"Surge u: {self.u:.2f} m/s",
            f"Sway v: {self.v:.2f} m/s",
            f"Yaw rate r: {self.r:.2f} rad/s",
            "",
            f"Waypoint: {waypoint_information}"
        ]

        # Draw every information line
        for line_index, text in enumerate(
            information_lines
        ):

            text_surface = self.font.render(
                text,
                True,
                config.TEXT_COLOR
            )

            self.screen.blit(
                text_surface,
                (
                    panel_x,
                    panel_y
                    + line_index * line_spacing
                )
            )

    def draw_usv(self):

        # Convert the USV center position to screen pixels
        center_x, center_y = (
            self.world_to_screen(
                self.x,
                self.y
            )
        )

        # USV drawing size in pixels
        usv_length = 42.0
        usv_width = 22.0

        # USV triangle points before rotation
        local_points = [
            (
                usv_length / 2,
                0
            ),
            (
                -usv_length / 2,
                -usv_width / 2
            ),
            (
                -usv_length / 2,
                usv_width / 2
            )
        ]

        rotated_points = []

        cos_psi = math.cos(
            self.psi
        )

        sin_psi = math.sin(
            self.psi
        )

        # Rotate every USV point around the USV center
        for local_x, local_y in local_points:

            rotated_x = (
                local_x * cos_psi
                - local_y * sin_psi
            )

            rotated_y = (
                local_x * sin_psi
                + local_y * cos_psi
            )

            screen_point = (
                center_x + rotated_x,
                center_y + rotated_y
            )

            rotated_points.append(
                screen_point
            )

        # Draw the USV body
        pygame.draw.polygon(
            self.screen,
            config.USV_COLOR,
            rotated_points
        )

        # Draw the USV outline
        pygame.draw.polygon(
            self.screen,
            config.USV_OUTLINE_COLOR,
            rotated_points,
            3
        )

        # Length of the heading indicator line
        heading_line_length = 35

        # Calculate the heading line end position
        heading_end_x = (
            center_x
            + heading_line_length * cos_psi
        )

        heading_end_y = (
            center_y
            + heading_line_length * sin_psi
        )

        # Draw the heading direction
        pygame.draw.line(
            self.screen,
            config.CURRENT_WAYPOINT_COLOR,
            (
                center_x,
                center_y
            ),
            (
                heading_end_x,
                heading_end_y
            ),
            3
        )

    def render(self):

        # Draw the complete simulation frame
        self.draw_ocean()
        self.draw_grid()
        self.draw_waypoints()
        self.draw_usv()
        self.draw_panel()

        # Display the completed frame
        pygame.display.flip()

        # Limit the display speed
        self.clock.tick(
            config.FPS
        )

    def close(self):

        # Close the Pygame window
        pygame.quit()