# Pygame display
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 720

# Right-side information panel
PANEL_WIDTH = 280

# The remaining area is used to display the ocean
OCEAN_WIDTH = SCREEN_WIDTH - PANEL_WIDTH
OCEAN_HEIGHT = SCREEN_HEIGHT

FPS = 60

# Convert physical distance to screen pixels
PIXELS_PER_METER = 8.0

# Simulation time step in seconds
DT = 0.05

# Maximum number of simulation steps in one episode
MAX_STEPS = 5000

# Simplified simulation parameters
# These are not yet the exact Otter USV parameters
MASS = 50.0
YAW_INERTIA = 25.0
SURGE_DAMPING = 12.0
SWAY_DAMPING = 20.0
YAW_DAMPING = 8.0

# Control limits
MAX_SURGE_FORCE = 80.0
MAX_YAW_MOMENT = 40.0
MAX_SURGE_SPEED = 4.0
MAX_SWAY_SPEED = 2.0
MAX_YAW_RATE = 1.5

# Waypoint settings
NUMBER_OF_WAYPOINTS = 5
WAYPOINT_RADIUS = 2.5
MIN_WAYPOINT_DISTANCE = 12.0
WAYPOINT_MARGIN = 8.0

# Disturbance settings
MAX_CURRENT_SPEED = 0.8
MAX_WIND_FORCE = 15.0
MAX_WAVE_MOMENT = 8.0
GUST_PROBABILITY = 0.005

# UI colors
OCEAN_COLOR = (28, 120, 170)
GRID_COLOR = (42, 140, 185)
USV_COLOR = (245, 245, 245)
USV_OUTLINE_COLOR = (25, 40, 55)
CURRENT_WAYPOINT_COLOR = (255, 210, 40)
FUTURE_WAYPOINT_COLOR = (240, 90, 70)
COMPLETED_WAYPOINT_COLOR = (80, 220, 130)
REFERENCE_PATH_COLOR = (210, 225, 235)
TRAJECTORY_COLOR = (255, 255, 255)
PANEL_COLOR = (20, 31, 45)
TEXT_COLOR = (235, 240, 245)
