from typing import Optional
import numpy as np
from scipy import interpolate
import cv2
from gymnasium import spaces
# BBO: A base environment implementing basic discounting and reward calculation
from envs.bbo import BBO

from collections import namedtuple
# Named tuple for storing rendered image dimensions
ImgDim = namedtuple('ImgDim', 'width height')

class Shape(BBO):
    """
    A 2D shape-optimization environment.
    
    State:
      A flat vector of N = state_dim knot values, arranged as a square grid (sqrt(N) x sqrt(N)).
      These knot values define a 2D surface via bicubic spline interpolation. 
      (spline is a smooth curve that goes through certain control points. Bicubic spline interpolation means it uses cubic polynomials in both the x and y directions)
        
    Action:
      A vector of the same length (state_dim), representing small additive adjustments to each knot value.

    Objective:
      Build a closed contour (loop, forming a boundary with np gaps) from the interpolated surface 
      (threshold → binary image → extract contour), then minimize perimeter / sqrt(area) of that contour.

    Inherits from BBO, which provides:
      - naive vs. shaped reward calculation
      - discounting logic
      - RNG seeding and episode-length tracking
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, naive=False, step_size=1e-2, state_dim=64, max_num_step=20, render_mode='human'):
        # Initialize the base BBO environment
        #  - naive: if True, use simple reward = -val. else use pmp-shaped reward
        #  - step_size: scaling factor how much each action perturbs the step
        #  - max_num_step: maximum number of steps before episode terminates
        super(Shape, self).__init__(naive, step_size, max_num_step)

        # State and action info 
        
        # State is a 1D vector that holds values for all control points (knots) in the grid
        # State_dim tells us how many of these values there are. Must be a perfect square
        self.state_dim = state_dim
        # Allowed range for each state dimension: [-4, 4]
        self.max_val = 4; self.min_val = -4
        # Allowed range for each action dimension: [-1, 1]
        self.max_act = 1; self.min_act = -1
        # Tells gym environment that action is a vector of state_dim values, each between -1 and 1
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)
        # Tells gym environment that observation (state) is a vector of state_dim numbers, where each number is between -4 and 4
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
        # Will eventually hold current state vector (current shape configuration). Initialized as None and will be set during the reset() function
        self.state = None
        # Determines how rendering works. human = show shape in a window. rgb_array returns pixel data instead
        self.render_mode = render_mode

        # Shape interpolation info
        # Creates an 8x8 grid of points from -1 to 1 in both x and y directions. These are knot positions. State vector provide values at each of these points
        self.xk, self.yk = np.mgrid[-1:1:8j, -1:1:8j]
        # Creates a 50x50 grid covering the same area, but much finer. Bicubic spline is evaluated here to get a smooth surface. 
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        # Stores the dimensions of the fine grid so we know how big the rendered image will be. 
        self.viewer = ImgDim(width=self.xg.shape[0], height=self.yg.shape[1])

    def step(self, action):
        """
        1. Apply continuous action to update knots:
             state_new = state_old + step_size * action
        2. Increment step counter
        3. Compute contour area & perimeter via helper:
             a. Bicubic spline → binary image → contour extraction
        4. Check for degeneration (zero area or perimeter)
        5. Compute objective val = peri / sqrt(area) or large penalty if degenerate
        6. Compute reward via calculate_final_reward(val, action)
        7. Return (next_state, reward, done, truncated=False, info={})
        """
        # 1. Update knots
        self.state += self.step_size *action

        # 2. Increment step counter
        self.num_step += 1

        # 3. Current knot values are converted into a smooth shape using bicubic spline interpolation.
        #    That shape is turned into a binary image, and then the area and perimeter of its boundary are computed.
        area, peri = geometry_info(self.state, self.xk, self.yk, self.xg, self.yg)
        
        # 4. If the shape has no area or no edge length, episode ends
        done = (area == 0 or peri == 0)
        
        # Reward Computation
        if not done:
            # Value to optimize: perimeter / sqrt(area)
            val = peri/np.sqrt(area)
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
        """
        Reset environment to start a new episode:
          - Reset RNG and discount via BBO.reset()
          - Zero step counter and discount
          - Initialize knot values via reset_at(mode='random')
        """
        # Calls the parent class BBO's reset method
        super().reset(seed=seed)
        # Resets the number of steps taken in the episode back to 0
        self.num_step = 0
        # Resets the discount multiplier for reward shaping
        self.discount = 1.0
        # Sets the shape's control points (knots) using random shape
        return self.reset_at(mode='random'), {}
    
    def reset_at(self, mode='random'):
        """
        Initialize knot values according to:
          - 'hole': central square of zeros + noise, clipped [0,1]
          - 'random': uniform random in [0,1)
          - 'random_with_padding': random interior, border stays at 1
        After generation, shift values to [-0.5, 0.5] and flatten.
        """

        # Resets the number of steps taken in the episode back to 0
        self.num_step = 0
        # Side length of square grid = sqrt(state_dim)
        width = int(np.sqrt(self.state_dim))
        # Represents a fully filled shape that is full of ones
        self.state = np.ones((width, width))
        
        if mode=='hole':
            # Set the center of the grid to 0, making a square hole
            self.state[1:8, 1:8] = 0
            # Add small random noise over entire grid
            self.state += self.rng.random((width, width))
            # Ensure values remain in [0, 1]
            self.state = np.clip(self.state, 0, 1)
            
        elif mode=='random':
            # Fully random values in [0, 1)
            self.state = self.rng.random((width, width))
            
        elif mode=='random_with_padding':
            # Fill the inner part of the grid with random values. Leave the outermost top and right edges as 1 (unchanged form initialization)
            self.state[1:(width-1), :(width-1)] = self.rng.random((width-2, width-1))
        
        # Shift range [0, 1] -> [-0.5, 0.5] around zero
        self.state -= .5
        # Environment expects a 1D vector, so we flatten 2D grid into a long array
        self.state = self.state.reshape(-1)
        
        # Returns the new state as a NumPy array so the agent can start acting
        return np.array(self.state)
    
    def render(self):
        """
        Turns the current state (a grid of shape values) into a black and white image that 
        visually shows the shape. It performs smoothing, binarization, and inversion to make
        it suitable for display or evaluation
        """
        # xk, yk: the coarse knot grid (where the shape values live)
        # xg, yg: the fine grid (where we evaluate the smooth version)
        xk, yk, xg, yg = self.xk, self.yk, self.xg, self.yg
        # shape was stored as a 1D array. self.state.reshape(xk.shape) turns it back into a 2D shape 
        # spline_interp() takes the knot grid and computes a smooth surface over the fine grid. Then it
        # clips the smoothed values to stay within [-1, 1], scales to grayscale pixel values (0-255), and
        # thresholds to make a binary age (shape = white, background = black)
        # 255 - binary_img makes the shape white and the background black.
        return 255-spline_interp(self.state.reshape(xk.shape[0], yk.shape[0]), xk, yk, xg, yg)
    
    def close(self):
        """
        Clean up any rendering resources.
        Here we simply drop the viewer reference.
        """
        if self.viewer:
            self.viewer = None

## Helper functions ##
# Spline interpolation for 2D density problem
def spline_interp(z, xk, yk, xg, yg):
    """
    Converts a 2D grid of shape values into a smooth black and white image using bicubic spline
    interpolation, then thresholds it to produce a clear contour of the shape.
    """
    # Create a smooth spline from the coarse shape grid. 
    # z is the grid of shape heights (the state reshaped). xk, yk are the 8x8 coordinate grids.
    tck = interpolate.bisplrep(xk, yk, z)
    # Evaluate the spline on a 50x50 grid (xg, yg). zint is now a smooth 50x50 float image, where values go from -1 to 1
    zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)
    # zint is between [-1, 1]
    zint = np.clip(zint, -1, 1)
    # Convert spline values to binary image
    C = 255/2; thresh = C
    # Convert to grayscale image. Transform -1 -> 0 (black), 0 -> 128 (gray), 1 -> 255 (white). 
    img = np.array(zint*C+C).astype('uint8')
    # Threshold to black and white. Pixels above 127.5 -> 255 (white). Pixels below -> 0 (black)
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return thresh_img

def geometry_info_from_img(img):
    """
    Given a black and white image, compute the total area of all shapes in white and the total perimeter of all those shapes
    """
    # Uses OpenCV to trace all white shapes in the image. Each shape becomes a list of (x, y) points around its border.
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # initialize area and perimeter
    area = 0; peri = 0
    # Loop through all detected contours. cnt is a single shape's outline
    for cnt in contours:
        # Gives signed area: it's positive for counter-clockwise contours (outer shapes) and negative for clockwise ones (holes)
        # - sign flips it so outer shapes contribute positively and holes subtract correctly
        area -= cv2.contourArea(cnt, oriented=True)
        # Measures total length around the shape
        peri += cv2.arcLength(cnt, closed=True)
    
    return area, peri

def geometry_info(z, xk, yk, xg, yg):
    """
    Turns a 2D array of shape values into a clean black-and-white image. 
    measures the area and perimeter of the shape in that image
    """
    # Takes the 2D array z. Uses bicubic spline interpolation to turn it into a smooth 50x50 image. Then threshold it into a binary image. 
    img = spline_interp(z, xk, yk, xg, yg)
    #Find all contours in the image. Computes total area and perimeter of the visible shape. Returns them as a tuple: (area, perimeter)
    return geometry_info_from_img(img)