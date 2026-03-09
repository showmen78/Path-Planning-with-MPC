import pygame
import numpy as np
import collections
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ==========================================
# 1. Configuration & Tuning Parameters
# ==========================================
# (These match the "Constants" from the guide)
SCREEN_SIZE = (1200, 800)
FPS = 60
SIMULATION_SCALE = 10.0  # 1 meter = 10 pixels

# Physical Limits (Manual Tuning Constants)
A_MAX_BRAKING = 8.0  # Max Deceleration (m/s^2) - defines Red Zone
A_N_COMFORT = 2.0     # Comfort Deceleration (m/s^2) - defines Green Zone
T0_REACTION = 1.0     # System Reaction Time (s)
SHAPE_EXPONENT = 4    # 2=Ellipse, 4=Rounded Rectangle (Squircle)
INTENSITY_A = 20.0    # Cost Scaling Factor
D_SAFE_BUFFER = 0.5   # Static buffer (m)

# Colors
COLOR_BACKGROUND = (30, 30, 30)
COLOR_TEXT = (220, 220, 220)
COLOR_EGO = (0, 100, 255)
COLOR_OBSTACLE = (200, 50, 50)
COLOR_SAFE_ZONE = (0, 200, 0, 100)  # Translucent Green
COLOR_COLLISION_ZONE = (200, 0, 0, 100) # Translucent Red
COLOR_GRID = (50, 50, 50)

# Collision States (Graph Output)
STATE_CLEAR = -1
STATE_SAFE_BOUNDARY = 0
STATE_SAFE_ZONE = 1
STATE_COLLISION_ZONE = 2

# Initialize Pygame
pygame.init()
pygame.display.set_caption("Autonomous Vehicle Potential Field Visualization")
screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# ==========================================
# 2. Vehicle Class Definition
# ==========================================
class Vehicle:
    def __init__(self, x, y, length, width, angle=0, color=COLOR_OBSTACLE, speed=0):
        """Physical units in METERS and RADIANS."""
        self.x = x  # meters
        self.y = y  # meters
        self.length = length  # meters
        self.width = width  # meters
        self.angle = angle  # radians
        self.color = color
        self.speed = speed  # m/s

    def update(self, dt, run=0):
        """Basic constant velocity movement."""
        self.x += self.speed * np.cos(self.angle) * dt *run #run make sure the vehicle only moves when key pressed
        self.y += self.speed * np.sin(self.angle) * dt * run

    def get_bounding_box(self):
        """Converts physical meters to screen pixels and handles rotation."""
        l_px, w_px = self.length * SIMULATION_SCALE, self.width * SIMULATION_SCALE
        x_px, y_px = self.x * SIMULATION_SCALE, self.y * SIMULATION_SCALE

        # Create rectangle, rotate, and get centered rect
        surface = pygame.Surface((l_px, w_px), pygame.SRCALPHA)
        pygame.draw.rect(surface, self.color, (0, 0, l_px, w_px))
        rotated_surface = pygame.transform.rotate(surface, -np.degrees(self.angle))
        rect = rotated_surface.get_rect(center=(x_px, y_px))
        return rotated_surface, rect

    def draw(self, screen):
        """Draws the vehicle surface on the screen."""
        surface, rect = self.get_bounding_box()
        screen.blit(surface, rect)

class EgoVehicle(Vehicle):
    def __init__(self, x, y, length, width, angle=0):
        super().__init__(x, y, length, width, angle, COLOR_EGO, speed=0)
        # Ego-specific controls (acceleration/steering constants)
        self.acceleration_limit = 15.0  # m/s^2 (tuning parameter)
        self.steering_speed = 3.0  # rad/s (tuning parameter)

    def handle_input(self, keys, dt):
        """Keyboard control logic."""
        if keys[pygame.K_UP]:
            self.speed += self.acceleration_limit * dt
        if keys[pygame.K_DOWN]:
            self.speed -= self.acceleration_limit * dt
        if keys[pygame.K_LEFT]:
            self.angle -= self.steering_speed * dt
        if keys[pygame.K_RIGHT]:
            self.angle += self.steering_speed * dt

# ==========================================
# 3. Math & Logic Functions
# ==========================================
def get_collision_state(ego, obs):
    """
    Implements the Master Guide to Potential Field Calculation.
    Returns: Current State (-1 to 2) and rc (normalized collision distance).
    """
    # Step 3: Coordinate Transformation (Handle Orientation)
    # (Move Ego relative to Obstacle Center $(x_o, y_o)$ and Heading $(\psi_o)$)
    dx = ego.x - obs.x
    dy = ego.y - obs.y
    x_local = dx * np.cos(obs.angle) + dy * np.sin(obs.angle)
    y_local = -dx * np.sin(obs.angle) + dy * np.cos(obs.angle)

    # Step 4: Calculate Relative Approach Velocities
    heading_diff = ego.angle - obs.angle
    v_approach_longitudinal = ego.speed * np.cos(heading_diff) - obs.speed
    v_approach_lateral = ego.speed * np.sin(heading_diff)

    delta_u = max(0, v_approach_longitudinal)
    delta_v = max(0.1, abs(v_approach_lateral))  # Avoid divide by zero

    # # Step 5: Static Inflation (Handle Vehicle Sizes)
    # x0 = (ego.length + obs.length) / 2 + D_SAFE_BUFFER
    # y0 = (ego.width + obs.width) / 2 + D_SAFE_BUFFER

    # Step 5: Advanced Dynamic Inflation (Handles Ego Rotation)
    heading_diff = ego.angle - obs.angle
    
    # Project Ego's physical dimensions onto the Obstacle's local X and Y axes
    projected_ego_length = abs(ego.length * np.cos(heading_diff)) + abs(ego.width * np.sin(heading_diff))
    projected_ego_width = abs(ego.length * np.sin(heading_diff)) + abs(ego.width * np.cos(heading_diff))
    
    # Calculate the inflated bounding box
    x0 = (projected_ego_length + obs.length) / 2 + D_SAFE_BUFFER
    y0 = (projected_ego_width + obs.width) / 2 + D_SAFE_BUFFER

    # Step 6: Calculate Dynamic Boundaries (Physics)
    xc = x0 + (delta_u**2) / (2 * A_MAX_BRAKING)
    yc = y0 + (delta_v**2) / (2 * A_MAX_BRAKING)

    xs = x0 + (delta_u * T0_REACTION) + (delta_u**2) / (2 * A_N_COMFORT)
    ys = y0 + (delta_v * T0_REACTION) + (delta_v**2) / (2 * A_N_COMFORT)

    # Step 7: Super-Ellipse Distance Metric (r) - The normalized scaling formula
    n = SHAPE_EXPONENT
    rc = (np.abs(x_local / xc)**n + np.abs(y_local / yc)**n)**(1/n)
    rs = (np.abs(x_local / xs)**n + np.abs(y_local / ys)**n)**(1/n)

    # Step 8: State Assignment (Visualizing the Field for Graphing)
    if rc <= 1.0:
        return STATE_COLLISION_ZONE, rc, (xc, yc, xs, ys)
    if rs <= 1.0:
        return STATE_SAFE_ZONE, rc, (xc, yc, xs, ys)
    if rs <= 1.01: # Small buffer for "On Boundary"
        return STATE_SAFE_BOUNDARY, rc, (xc, yc, xs, ys)

    return STATE_CLEAR, rc, (xc, yc, xs, ys)

def draw_boundaries(screen, obs, boundaries_data):
    """Draws rotated Translucent Super-Ellipses."""
    # We unpack the math boundaries, but we will adjust them for drawing
    xc_math, yc_math, xs_math, ys_math = boundaries_data

    # To make the visual match the physical bumper-to-bumper collision, 
    # we must subtract the Ego's inflation from the drawing boundaries!
    # (Assuming average Ego dimensions if projecting is too complex for drawing)
    ego_l_inflation = ego.length / 2 + D_SAFE_BUFFER
    ego_w_inflation = ego.width / 2 + D_SAFE_BUFFER

    xc = max(0.1, xc_math - ego_l_inflation)
    yc = max(0.1, yc_math - ego_w_inflation)
    xs = max(0.1, xs_math - ego_l_inflation)
    ys = max(0.1, ys_math - ego_w_inflation)

    # Make the drawing surface large enough to hold the field
    l_px, w_px = (xs * 2 + 10) * SIMULATION_SCALE, (ys * 2 + 10) * SIMULATION_SCALE
    field_surface = pygame.Surface((l_px, w_px), pygame.SRCALPHA)
    
    grid_size = 300
    gx = np.linspace(-l_px/(2*SIMULATION_SCALE), l_px/(2*SIMULATION_SCALE), grid_size)
    gy = np.linspace(-w_px/(2*SIMULATION_SCALE), w_px/(2*SIMULATION_SCALE), grid_size)
    GX, GY = np.meshgrid(gx, gy)

    n = SHAPE_EXPONENT
    
    # Calculate grids of 'r' values to create mask (using the visual boundaries)
    rc_grid = (np.abs(GX / xc)**n + np.abs(GY / yc)**n)**(1/n)
    rs_grid = (np.abs(GX / xs)**n + np.abs(GY / ys)**n)**(1/n)

    # Create masks
    collision_mask = (rc_grid <= 1.0).astype(int)
    safe_mask = ((rs_grid <= 1.0) & (rc_grid > 1.0)).astype(int)

    # Create pixel arrays
    red_field = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
    red_field[collision_mask == 1] = list(COLOR_COLLISION_ZONE)
    
    green_field = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
    green_field[safe_mask == 1] = list(COLOR_SAFE_ZONE)

    combined_field = red_field + green_field

    # Convert to Pygame surface
    boundary_surf = pygame.image.frombuffer(combined_field.flatten(), (grid_size, grid_size), 'RGBA')
    scaled_surf = pygame.transform.scale(boundary_surf, (l_px, w_px))
    rotated_surf = pygame.transform.rotate(scaled_surf, -np.degrees(obs.angle))
    new_rect = rotated_surf.get_rect(center=(obs.x * SIMULATION_SCALE, obs.y * SIMULATION_SCALE))

    screen.blit(rotated_surf, new_rect)

# ==========================================
# 4. Matplotlib Graphing Class
# ==========================================
class RealTimeGraph:
    def __init__(self, x_px, y_px, width_px, height_px, max_points=200):
        self.rect = pygame.Rect(x_px, y_px, width_px, height_px)
        self.max_points = max_points
        self.data_state = collections.deque(maxlen=max_points)
        self.data_time = collections.deque(maxlen=max_points)
        self.start_time = pygame.time.get_ticks()

    def update(self, state):
        current_time = (pygame.time.get_ticks() - self.start_time) / 1000.0  # seconds
        self.data_state.append(state)
        self.data_time.append(current_time)

    def draw(self, screen):
        if len(self.data_state) < 2:
            return

        # Render Matplotlib Figure
        fig = Figure(figsize=(self.rect.width / 100.0, self.rect.height / 100.0), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#222222') # Match screen backround
        ax.set_facecolor('#333333')

        times = list(self.data_time)
        states = list(self.data_state)

        # Plot settings
        ax.plot(times, states, color='#FFDD00', linewidth=2)
        ax.set_ylim(-1.5, 2.5)
        ax.set_yticks([-1, 0, 1, 2])
        ax.set_yticklabels(['Clear', 'On Bndry', 'Safe Zone', 'Coll Zone'], color='white')
        ax.set_xticklabels([])
        ax.set_title("Worst-Case Collision State", color='white', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5, color='gray')

        # Convert to Pygame surface
        # Convert to Pygame surface (Updated for Matplotlib 3.8+)
        canvas = FigureCanvas(fig)
        canvas.draw()
        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(canvas.buffer_rgba(), size, "RGBA")

        # Display
        screen.blit(surf, self.rect)

# ==========================================
# 5. Main Simulation Initialization
# ==========================================
# (x, y, l, w, angle_rad, speed)
ego = EgoVehicle(x=20.0, y=40.0, length=4.5, width=2.0)

# Two obstacles with different shapes and orientations
obs1 = Vehicle(x=80.0, y=20.0, length=8.0, width=3.0, angle=np.radians(20), speed=0.0) # Truck
obs2 = Vehicle(x=50.0, y=60.0, length=4.0, width=2.0, angle=np.radians(-10), speed=0.0) # Car

vehicles = [obs1, obs2]
graph = RealTimeGraph(x_px=800, y_px=50, width_px=380, height_px=300)

# ==========================================
# 6. Main Game Loop
# ==========================================
running = True
dt = 0.0
run = 0

while running:
    # --- Input Handling ---
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            run = 1
            if event.key == pygame.K_r: # Reset Simulation
                ego = EgoVehicle(x=20.0, y=40.0, length=4.5, width=2.0)
                graph = RealTimeGraph(x_px=800, y_px=50, width_px=380, height_px=300)

        else: run =0

    # --- Physics Update ---
    # Convert ticks (ms) to time delta (seconds)
    dt = clock.tick(FPS) / 1000.0  
    ego.handle_input(keys, dt)
    ego.update(dt, run)
    for obs in vehicles:
        obs.update(dt)

    # --- Collision & Potential Field Logic (Calculated Real-Time) ---
    highest_state = STATE_CLEAR
    highest_rc = float('inf')
    boundaries_draw_data = []

    # Iterate through obstacles to calculate fields and worst-case danger state
    for obs in vehicles:
        state, rc, bnd_data = get_collision_state(ego, obs)
        boundaries_draw_data.append((obs, bnd_data))
        if state > highest_state:
            highest_state = state
            highest_rc = rc
        elif state == highest_state:
            # If same state, track the closest (lowest normalized distance)
            highest_rc = min(highest_rc, rc)

    # Update Graph
    graph.update(highest_state)

    # --- Drawing ---
    screen.fill(COLOR_BACKGROUND)
    
    # Draw simple background grid for scale perspective (every 5m)
    grid_gap_px = int(5.0 * SIMULATION_SCALE)
    for x in range(0, SCREEN_SIZE[0], grid_gap_px):
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, SCREEN_SIZE[1]))
    for y in range(0, SCREEN_SIZE[1], grid_gap_px):
        pygame.draw.line(screen, COLOR_GRID, (0, y), (SCREEN_SIZE[0], y))

    # Visualize Potential Field Boundaries FIRST (behind vehicles)
    for obs, bnd_data in boundaries_draw_data:
        draw_boundaries(screen, obs, bnd_data)

    # Draw Vehicles
    ego.draw(screen)
    for obs in vehicles:
        obs.draw(screen)

    # Draw Real-Time Graph
    graph.draw(screen)

    # Visualize HUD Text (Top Left)
    text_y = 10
    hud_texts = [
        ("Autonomous Vehicle Control Sim", COLOR_TEXT),
        ("", COLOR_TEXT),
        (f"Controls:", COLOR_TEXT),
        ("ARROW KEYS: Move/Steer Ego", COLOR_EGO),
        ("R: Reset Simulation", COLOR_TEXT),
        ("", COLOR_TEXT),
        (f"Ego State:", COLOR_EGO),
        (f"  Speed: {ego.speed:.1f} m/s", COLOR_EGO),
        (f"  Heading: {np.degrees(ego.angle):.1f}°", COLOR_EGO),
        ("", COLOR_TEXT),
        (f"Potential Field (Worst Case):", COLOR_OBSTACLE),
        (f"  Danger State: {highest_state}", COLOR_OBSTACLE),
        (f"  Danger Ratio (rc): {highest_rc:.2f}", COLOR_OBSTACLE),
    ]
    for text, color in hud_texts:
        img = font.render(text, True, color)
        screen.blit(img, (10, text_y))
        text_y += 22

    # Refresh Screen
    pygame.display.flip()

# Clean exit
pygame.quit()