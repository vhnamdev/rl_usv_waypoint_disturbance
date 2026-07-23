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
        self.font = pygame.font.SysFont("arial",22)

        # USV position in meters
        self.x = 40.0
        self.y = 40.0

        # Heading angle in radians
        self.psi = 0.0

        # Velocities
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def world_to_screen(self,x_in_meters, y_in_meters):

        screen_x = int(x_in_meters*config.PIXELS_PER_METER)
        screen_y = int(y_in_meters*config.PIXELS_PER_METER)

        return screen_x,screen_y

    def draw_ocean(self):

        ocean_rect = pygame.Rect(
            0,0, config.OCEAN_WIDTH, config.OCEAN_HEIGHT
        )

        pygame.draw.rect(self.screen, config.OCEAN_COLOR, ocean_rect)

    def draw_grid(self):

        grid_spacing_meters = 5.0

        grid_spacing_pixels = int(grid_spacing_meters*config.PIXELS_PER_METER)

        for x_pixel in range(0,config.OCEAN_WIDTH,grid_spacing_pixels):
            pygame.draw.line(self.screen,config.GRID_COLOR,(x_pixel,0),(x_pixel,config.OCEAN_HEIGHT),1)

        for y_pixel in range(0,config.OCEAN_HEIGHT,grid_spacing_pixels):
            pygame.draw.line(self.screen,config.GRID_COLOR,(0,y_pixel),(config.OCEAN_WIDTH,y_pixel),1)

    def draw_panel(self):

        panel_rect = pygame.Rect(config.OCEAN_WIDTH,0,config.PANEL_WIDTH,config.SCREEN_HEIGHT)

        pygame.draw.rect(self.screen,config.PANEL_COLOR,panel_rect)

        panel_x = config.OCEAN_WIDTH + 25
        panel_y = 35
        line_spacing = 34

        information_lines = [ "USV INFORMATION",
            "",
            f"Position X: {self.x:.2f} m",
            f"Position Y: {self.y:.2f} m",
            f"Heading: {math.degrees(self.psi):.2f} deg",
            f"Surge u: {self.u:.2f} m/s",
            f"Sway v: {self.v:.2f} m/s",
            f"Yaw rate r: {self.r:.2f} rad/s"
            ]

        for line_index, text in enumerate(information_lines):
            text_surface = self.font.render(text,True,config.TEXT_COLOR)
            self.screen.blit(text_surface,(panel_x,panel_y + line_index*line_spacing)) 

    def draw_usv(self):
        
        center_x,center_y = self.world_to_screen(self.x,self.y)

        usv_length = 42.0
        usv_width = 22.0

        local_points = [(usv_length / 2,0),(-usv_length / 2,-usv_width / 2),(-usv_length / 2, usv_width / 2)]

        rotated_points = []

        cos_psi = math.cos(self.psi)
        sin_psi = math.sin(self.psi)

        for local_x, local_y in local_points:
            rotated_x = (local_x * cos_psi - local_y * sin_psi)
            rotated_y = (local_x * sin_psi + local_y * cos_psi)
            screen_point = (center_x + rotated_x, center_y + rotated_y)
            rotated_points.append(screen_point)

        pygame.draw.polygon(self.screen,config.USV_COLOR,rotated_points)

        pygame.draw.polygon(self.screen,config.USV_OUTLINE_COLOR,rotated_points,3)

        heading_line_length = 35

        heading_end_x = (center_x + heading_line_length * cos_psi)
        heading_end_y = (center_y + heading_line_length * sin_psi)

        pygame.draw.line(self.screen,config.CURRENT_WAYPOINT_COLOR,(center_x,center_y),(heading_end_x,heading_end_y),3)

    def render(self):
        self.draw_ocean()
        self.draw_grid()
        self.draw_usv()
        self.draw_panel()

        pygame.display.flip()

        self.clock.tick(config.FPS)


    def close(self):
        pygame.quit()


if __name__ == "__main__":
    environment = USVEnvironment()

    while environment.running:
        environment.handle_events()
        environment.render()

    environment.close()