from typing import Optional
import numpy as np
from shapely.geometry import Polygon
from gymnasium import spaces
import pygame
from pygame import gfxdraw
# BBO: A base environment implementing basic discounting and reward calculation
from envs.bbo import BBO
import splinepy as sp

# A large numeric constant used to penalize degenerate shapes
MAX_ACT = 1e4

class ShapeBoundary(BBO):
    metadata = {
        # Supported rendering modes: interactive human window or RGB array for video
        "render_modes": ["human", "rgb_array"],
        # Framers per second when rendering
        "render_fps": 15,
    }

    def __init__(self, naive=False, step_size=1e-2, ctrl_state_dim=16, max_num_step=20, render_mode='human', degree=3, n_internal_knots=4):
        # Initialize the base BBO environment
        #  - naive: if True, use simple reward = -val. else use pmp-shaped reward
        #  - step_size: scaling factor how much each action perturbs the step
        #  - max_num_step: maximum number of steps before episode terminates
        super(ShapeBoundary, self).__init__(naive, step_size, max_num_step)

        # ——— spline + knot control parameters ———
        self.ctrl_dim        = ctrl_state_dim                   # original 16 dims → 8 control‐points
        self.num_coef        = ctrl_state_dim // 2
        self.knot_dim        = n_internal_knots                 # e.g. 4 internal knots
        self.state_dim       = self.ctrl_dim + self.knot_dim    # total dims

        self.degree          = degree
        self.max_num_step    = max_num_step
        self.step_size       = step_size

        # redefine spaces with new state_dim
        self.observation_space = spaces.Box(-4, 4, (self.state_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1, 1, (self.state_dim,), dtype=np.float32)

        # spline sampling parameters
        self.ts   = np.linspace(0, 1, 80)
        self.verts = None
        
         # ——— Define initial & target circles ———
        # Build a *uniform open* knot vector of length = num_ctrl + degree + 1
        num_internal = self.num_coef - self.degree - 1
        # interior knots: i/(num_internal+1) for i=1..num_internal
        internal_knots = [i/(num_internal+1) for i in range(1, num_internal+1)]
        kv = [0.0]*(self.degree+1) + internal_knots + [1.0]*(self.degree+1)

        # radii (you can tweak these)
        R_big = 1.0      # unit circle
        R_small = 0.5    # half-radius circle

        # parametric angles for control points
        angles = np.linspace(0, 2*np.pi, self.num_coef, endpoint=False)

        # control points on big & small circles (2D)
        big_pts   = np.stack([np.cos(angles), np.sin(angles)], axis=1) * R_big
        small_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * R_small

        # store flat initial state + zero knot‐offsets
        self.initial_ctrl = big_pts.flatten()
        self.target_spline = sp.NURBS(
            degrees=[self.degree],
            knot_vectors=[kv],
            control_points=small_pts,
            weights=[1.0]*self.num_coef
        )

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 600
        self.screen = None # Pygame window or surface
        self.clock = None # Pygame clock for FPS control
        self.isopen = True # Tracks if the window is open
        
        # Spline specific paramters
        self.degree = degree  # Degree of the spline (default is cubic, degree=3)
        self.n_internal_knots = n_internal_knots  # Number of internal knots in the spline
    
    def _unpack_state(self):
        # first part = control‐point coords
        flat = self.state[:self.ctrl_dim]
        ctrl_pts = flat.reshape(self.num_coef, 2)

        # next = raw offsets → positive → simplex → increasing (0,1)
        if self.knot_dim > 0:
            raw = self.state[self.ctrl_dim:]
            exp = np.exp(raw - raw.max())      # stabilize exp
            probs = exp / exp.sum()
            knots_internal = np.cumsum(probs)
        else:
            knots_internal = []

        # build full open knot vector
        kv = [0.0]*(self.degree+1) + list(knots_internal) + [1.0]*(self.degree+1)
        return ctrl_pts, kv



    def step(self, action):
        # 1) Apply delta to full state (ctrl pts + knot offsets)
        self.state += self.step_size * action

        # 2) Build a splinepy NURBS from state
        ctrl_pts, kv = self._unpack_state()
        spline = sp.NURBS(
            degrees=[self.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
            weights=[1.0]*self.num_coef
        )

        # 3) Sample it and compute geometry
        coords  = spline.evaluate(self.ts.reshape(-1,1))   # shape (80,2)
        polygon = Polygon(zip(coords[:,0], coords[:,1]))

        # 4) Prepare verts for rendering
        scaled = coords / np.max(np.abs(coords)) * 100 + 300
        self.verts = list(zip(scaled[:,0], scaled[:,1]))

        # 5) Reward logic (same as before, but val based on isoperim or replace with your chamfer)
        done = (polygon.area == 0) or (self.num_step >= self.max_num_step)
        dist = self._distance(spline, self.target_spline)
        reward = -dist
        self.num_step += 1

        return self.state.copy(), reward, done, False, {}

    
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
    
    def reset_at(self, mode='unused'):
        """Always start on the big circle, no randomness."""
        self.num_step = 0

        # 1) control‐point portion = the big circle
        ctrl = self.initial_ctrl.copy()

        # 2) knot offsets start at zero
        if self.knot_dim > 0:
            self.state = np.concatenate([ctrl, np.zeros(self.knot_dim)])
        else:
            self.state = ctrl

        # 3) Precompute verts for rendering
        ctrl_pts, kv = self._unpack_state()
        spline = sp.NURBS([self.degree], [kv], ctrl_pts, weights=[1.0]*self.num_coef)
        coords = spline.evaluate(self.ts.reshape(-1,1))
        scaled = coords/np.max(np.abs(coords))*100 + 300
        self.verts = list(zip(scaled[:,0], scaled[:,1]))

        return self.state.copy()

    
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
            
    def _distance(self, a, b, samples=100):
        t = np.linspace(0,1,samples).reshape(-1,1)
        pa = a.evaluate(t)
        pb = b.evaluate(t)
        return np.mean(np.linalg.norm(pa - pb, axis=1))