from typing import Optional
import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon
from gymnasium import spaces
import pygame
from pygame import gfxdraw
# BBO: A base environment implementing basic discounting and reward calculation
from envs.bbo import BBO

# A large numeric constant used to penalize degenerate shapes
MAX_ACT = 1e4

class ShapeBoundary(BBO):
    metadata = {
        # Supported rendering modes: interactive human window or RGB array for video
        "render_modes": ["human", "rgb_array"],
        # Framers per second when rendering
        "render_fps": 15,
    }

    def __init__(self, naive=False, step_size=1e-2, state_dim=16, max_num_step=20, render_mode='human'):
        # Initialize the base BBO environment
        #  - naive: if True, use simple reward = -val. else use pmp-shaped reward
        #  - step_size: scaling factor how much each action perturbs the step
        #  - max_num_step: maximum number of steps before episode terminates
        super(ShapeBoundary, self).__init__(naive, step_size, max_num_step)

        # State and action info 
        # Number of parameters representing the shape boundary (state vector length). 
        # For example, if state_dim = 16, then the shape is controlled by 16 numbers, the first half representing x coordinates, and the second half representing y coordinates of control points for a cubic spline
        self.state_dim = state_dim 
        # Allowed range for each state dimension: [-4, 4]
        self.min_val = -4; self.max_val = 4 
        # Tells gym environment that observation (state) is a vector of state_dim numbers, where each number is between -4 and 4
        self.observation_space = spaces.box.Box(low=self.min_val, high=self.max_val, shape=(state_dim,), dtype=np.float32)
        # Allowed range for each action dimension: [-1, 1]
        self.min_act = -1; self.max_act = 1 
        # Tells gym environment that action is a vector of state_dim values, each between -1 and 1
        self.action_space = spaces.box.Box(low=self.min_act, high=self.max_act, shape=(state_dim,), dtype=np.float32)
        # Will eventually hold current state vector (current shape configuration). Initialized as None and will be set during the reset() function
        self.state = None 
    
        # Geometry
        # num_coef is how many control points (x, y) pairs there are. If state_dim = 16, then num_coef = 8, and we have 8 control points
        self.num_coef = self.state_dim//2
        # Creates 80 equally spaced numbers between 0 and 1. These are the parameter values used to evaluate the cubic spline. 
        self.ts = np.linspace(0, 1, 80)
        # Store (x, y) points after spline interpolation. These points are what get rendered on screen. Starts as none but is filled in later during reset() or step()
        self.verts = None

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 600
        self.screen = None # Pygame window or surface
        self.clock = None # Pygame clock for FPS control
        self.isopen = True # Tracks if the window is open

    def step(self, action):
        """
        Apply an action (delta changes to control points), update the state, compute the smoothed polygon, and return observation, reward, done.
        """
        # The environment state is incremented by step_size * action vector
        self.state += self.step_size * action

        # Creates a cubic spline function that smoothly passes through the control points
        # - np.linespace(0, 1, self.num)coef) is the evenly spaced times t for each control point
        # - self.state.reshape(2, self.num_coef).T) reshapes 1D list of numbers like [x1, x2, y1, y2] into [[x1, y1], [x2, y2]]
        cs = CubicSpline(np.linspace(0,1,self.num_coef), self.state.reshape(2, self.num_coef).T)
        # Uses spline to calculate 80 (x, y) points along the smooth curve. self.ts has 80 evenly spaced values between 0 and 1. 
        coords = cs(self.ts)
        
        # Zips the 80 x-values and y-values into (x, y) coordinate pairs. Then wraps them into a shapely.geometry.Polygon object
        # Polygon used for computing area, perimeter, or other geometry. Can also use polygon to later render the shape
        polygon = Polygon(zip(coords[:,0], coords[:,1]))
        
        # Prepare for Rendering
        # Normalize and scale coordinates into screen space [0, 600]^2
        # Center at (300, 300), scale so max absolute coordinate -> 100 units
        coords = coords/np.max(np.abs(coords))*100 + 300
        # Takes x-coordinates (coords[:,0]) and y-coordinates coords[:,1] and zips them into (x, y) point pairs. Saved to self.verts, later used when you draw shape on screen
        self.verts = list(zip(coords[:,0], coords[:,1]))
        
        # If the shape collapses (area = 0 or perimeter = 0), mark done
        done = (polygon.area == 0 or polygon.length == 0)
        
        # Increment internal step counter
        self.num_step += 1

        # Reward Computation
        if not done:
            # Value to optimize: perimeter / sqrt(area)
            val = polygon.length/np.sqrt(polygon.area)
            # Checks if the current step count has reached the limit
            done = self.num_step >= self.max_num_step
        else:
            # If shape was broken earlier, assign huge penalty value to indicate bad shape. Ensure very negative reward so agent learns to avoided these cases
            val = 1e9

        # Convert raw geometric value & action into final reward
        reward = self.calculate_final_reward(val, action)
            
        # Return: (observation, reward, done, truncated = False, info)
        return np.array(self.state), reward, done, False, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Calls the parent class BBO's reset method
        super().reset(seed=seed)
        # Resets the number of steps taken in the episode back to 0
        self.num_step = 0
        # Resets the discount multiplier for reward shaping
        self.discount = 1.0

        # Initializes state using half random preset, meaning half the x values are positive, half are negative, and y-values follow a simple ramp pattern. 
        # This makes the starting shape both non-trivial and diverse, which is good for learning
        return self.reset_at(mode='half_random'), {}
    
    def reset_at(self, mode='random'):
        """
        Initialize state vector to one of several modes
          - ellipse: control points form an ellipse
          - rect: control points form a rectangle
          - half_random: half positive random x, half negative random x, linear y
          - random: fully uniform random in [-0.5, 0.5]
        Then precompute self.verts from this initial state
        """
        # Resets the number of steps taken in the episode back to 0
        self.num_step = 0
        # Creates a zeroed out state vector that holds all control point coordinates. 
        self.state = np.zeros(self.state_dim)
        # Creates a linearly spaced array of num_coef values ranging from 0 to 1. Used in ellipse mode to assign evenly spaced angles or positions
        t = np.arange(self.num_coef)/self.num_coef
        
        if mode == 'ellipse':
            # x = 0.2 sin(2πt). Sine wave naturally gives circular/oval horizontal layout. 0.2 is a scaling factor, which makes ellipse narrower in x direction
            self.state[:self.num_coef] = 0.2*np.sin(2*np.pi*t)
            # y = cos(2πt). Cosine wave gives vertical component of ellipse. Since there's no scaling factor, y-axis will be taller than x-axis
            self.state[self.num_coef:] = np.cos(2*np.pi*t)
        elif mode == 'rect':
            # Total number of control points (num_coef) is split into 4 segments, one for each edge of the rectangle: top, right, bottom, left. n is how many points will be on each edge
            n = self.num_coef//4
            
            # x-coord
            # Top edge: x values go from 0 to 1 (evenly spaced). A horizontal line from left to right
            self.state[:n] = np.arange(n)/n
            # Right edge: x is always one. A vertical line going down to the right side
            self.state[n:2*n] = 1
            # Bottom edge: x values go from 1 -> 0. A horizontal line from right to left.
            self.state[2*n:3*n] = 1 - (np.arange(n)/n)
            # Left edge: x is always 0. A vertical line going up tot he left side. 
            self.state[3*n:4*n] = 0
            
            # y-coord
            # Top edge: y is 0. Flat along the bottom
            self.state[4*n:5*n] = 0
            # Right edge: y goes from 0 to 1. Vertical line rising on the right
            self.state[5*n:6*n] = np.arange(n)/n
            # Bottom edge: y is always 1. Horizontal line along the top
            self.state[6*n:7*n] = 1
            # Left edge: y goes from 1 to 0. Vertical line falling on the left
            self.state[7*n:8*n] = 1 - (np.arange(n)/n)
            
        elif mode == 'half_random':
            """
            Mode creates an initial shape that has a structured yet random layout, positive x values on one half of the shape, negative x-values on the other, and a simple clean layout in the y direction. 
            """
            # Divide total number of control points into two halves. Lets us apply different strategies to each half of the shape. 
            n = self.num_coef//2
            
            # x-coords
            # First half positive random in [0.2, 1.0]
            self.state[0:n] = 0.8*self.rng.random(n) + 0.2
            # Second half negative random in [-1.0, -0.2]
            self.state[n:2*n] = -0.8*self.rng.random(n) - 0.2
            
            #self.state[0:n] = 0.8*np.random.rand(n) + 0.2
            #self.state[n:2*n] = -0.8*np.random.rand(n) - 0.2

            # y-coord
            # For y coordinates, simple linear ramp in both halves
            self.state[2*n:3*n] = np.arange(n)/n
            self.state[3*n:4*n] = np.arange(n)/n
            
        elif mode == 'random':
            # Full random noise in [-0.5, 0.5]
            self.state = self.rng.random(self.state_dim) - 0.5
            
        # Converts flat state vector into a list of (x, y) control points. Then creates a cubic spline that passes through them smoothly.
        cs = CubicSpline(np.linspace(0,1,self.num_coef), self.state.reshape(2, self.num_coef).T)
        # Evaluates spline at 80 evenly spaced positions (self.ts). Gives you a smooth curve with 80 (x, y) points.
        coords = cs(self.ts)
        # Normalizes the coordinates so the largest absolute value becomes 100. Shifts all points so the shape is centered at (300, 300), middle of screen.
        coords = coords/np.max(np.abs(coords))*100 + 300
        # Converts the array of coordinates into a list of tuples: [(x1, y1), (x2, y2),...,(x80, y80)]. These are the vertices of the final shape that will be drawn in render()
        self.verts = list(zip(coords[:,0], coords[:,1]))
        # Returns the current state (shape parameters as a NumPy array). 
        return np.array(self.state)
    
    def render(self):
        """
        Draws the current shape defined by self.verts onto the screen using Pygame. It either shows it in a live window (human mode), or returns an image as an array (rgb_array mode)
        """
        # If the screen hasn't been created yet, initialize it
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                # If you're in human mode, start the display module, create a window of 600x600 pixels to draw on
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        # Create a clock to control the rendering frame rate. Only used in human mode so the animation runs smoothly. 
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Clear background to white
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        
        # Draw a clean black shape on a white background
        gfxdraw.aapolygon(self.surf, self.verts, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, self.verts, (0, 0, 0))
        
        # Pygame's default coordinate system has (0, 0) at the top-left. The shape is built with (0, 0) at the bottom left. 
        # This flips the drawing upside-down to match the match coordinates with screen coordinates
        self.surf = pygame.transform.flip(self.surf, False, True)
        
        # Takes the finished drawing on surf and puts it onto the visible screen
        self.screen.blit(self.surf, (0, 0))
        
        
        if self.render_mode == "human":
            # handles live rendering in a window, with proper frame timing
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            # Handles headless rendering where you want the pixels, not a live display
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
     
    def close(self):
        # Properly close the pygame window and quit
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False