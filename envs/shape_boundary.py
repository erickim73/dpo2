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

    def __init__(self, naive=False, step_size=1e-2, ctrl_state_dim=36, max_num_step=20, render_mode='human', degree=2, n_internal_knots=14, train_ctrl=True, train_weight=True, train_knot=True):
        # Initialize the base BBO environment
        #  - naive: if True, use simple reward = -val. else use pmp-shaped reward
        #  - step_size: scaling factor how much each action perturbs the step
        #  - max_num_step: maximum number of steps before episode terminates
        super(ShapeBoundary, self).__init__(naive, step_size, max_num_step)

        # spline + knot control parameters
        self.ctrl_dim        = ctrl_state_dim                   # e.g. 36 control points (2D×18)
        self.num_coef        = ctrl_state_dim // 2
        self.knot_dim        = n_internal_knots                 # e.g. 4 internal knots
        self.state_dim       = self.ctrl_dim + self.knot_dim    # total dims
        
        self.weight_dim = self.num_coef # one weight per control point
        # total state = [ctrl_pts (2D×num_coef) + weights + knot‐offsets]
        self.state_dim  = self.ctrl_dim + self.weight_dim + self.knot_dim

        # now redefine your spaces to match the new dimension:
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        self.degree          = degree
        self.max_num_step    = max_num_step
        self.step_size       = step_size
        
        self.learn_ctrl   = train_ctrl
        self.learn_weight = train_weight
        self.learn_knot   = train_knot

        # redefine spaces with new state_dim
        self.observation_space = spaces.Box(-4, 4, (self.state_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1, 1, (self.state_dim,), dtype=np.float32)

        # spline sampling parameters
        self.ts   = np.linspace(0, 1, 80)
        self.verts = None
        
        # Define initial & target circles
        # Build a *uniform open* knot vector using exactly n_internal_knots
        num_internal = self.knot_dim
        internal_knots = [i/(num_internal+1) for i in range(1, num_internal+1)]
        kv = [0.0] * (self.degree + 1) + internal_knots + [1.0] * (self.degree + 1)
        self.base_kv = kv
        # store the *uniform* internal knots so offsets are added to these
        self.base_internal = np.array(
            kv[self.degree+1 : - (self.degree+1)]
        )

        # Define improved "J" and target "E" shapes
        J_pts = np.array([
            [-0.1, 2.0],  # Top left of horizontal bar
            [0.6, 2.0],   # Top right of horizontal bar
            [0.6, 1.8],   # Top right corner
            [0.4, 1.8],   # Inner corner of bar
            [0.4, 1.6],   # Start of vertical descent
            [0.4, 1.3],   # Upper vertical
            [0.4, 1.0],   # Mid vertical
            [0.4, 0.7],   # Lower vertical
            [0.4, 0.4],   # Lower vertical before hook
            [0.4, 0.25],  # Hook transition
            [0.3, 0.1],   # Hook bend start
            [0.1, 0.0],   # Hook bend middle
            [-0.1, -0.05],# Bottom of hook
            [-0.3, 0.0],  # Left extent of hook
            [-0.4, 0.1],  # Hook left bottom curve
            [-0.4, 0.25], # Hook left side lower
            [-0.35, 0.35],# Hook left side upper
            [-0.25, 0.3]  # Hook closure
        ])

        E_pts = np.array([
            [-0.35, -0.05], # Bottom left corner - rounded
            [-0.4, 0.1],    # Left bottom curve
            [-0.4, 0.6],    # Left side lower
            [-0.4, 1.2],    # Left side middle
            [-0.4, 1.8],    # Left side upper
            [-0.4, 2.0],    # Top left corner
            [-0.1, 2.0],    # Top left inner
            [0.2, 2.0],     # Top line middle
            [0.4, 2.0],     # Top right end
            [0.4, 1.7],     # Top right corner
            [0.1, 1.7],     # Top line return
            [-0.2, 1.7],    # Top line to left edge
            [-0.2, 1.3],    # Left edge upper
            [0.2, 1.3],     # Middle line end
            [0.3, 1.0],     # Middle line right edge
            [-0.2, 1.0],    # Middle to left edge
            [-0.2, 0.3],    # Left edge lower
            [0.35, -0.05]   # Bottom right corner - rounded
        ])
        
        # Use weights to emphasize key structural points
        j_weights = [2.0, 2.5, 2.5, 2.0, 1.5, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5, 2.5, 2.0, 1.8, 1.5, 1.5]
        e_weights = [0.7, 1.0, 1.5, 1.5, 2.0, 2.5, 2.0, 2.0, 2.5, 2.5, 2.0, 1.5, 1.5, 2.0, 2.0, 1.5, 1.2, 0.7]

        
        # Create the target spline (E shape)
        self.target_spline = sp.NURBS(
            degrees=[self.degree],
            knot_vectors=[kv],
            control_points=E_pts,
            weights=e_weights
        )
        
        self.j_weights = j_weights
        # as a numpy array for easy math later
        self.initial_weights = np.array(self.j_weights, dtype=np.float32)

        # flatten for initial state (J shape)
        self.initial_ctrl = J_pts.flatten()

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
        self.base_internal = np.array(self.base_kv[self.degree+1 : - (self.degree+1)])
    
    def _unpack_state(self):
        # 1) control points
        flat    = self.state[:self.ctrl_dim]
        ctrl_pts = flat.reshape(self.num_coef, 2)

        # 2) weights
        w_slice = slice(self.ctrl_dim,
                        self.ctrl_dim + self.weight_dim)
        weights = self.state[w_slice]  # shape=(num_coef,)

        # 3) raw knot offsets
        k_slice = slice(self.ctrl_dim + self.weight_dim, None)
        raw_knot = self.state[k_slice] # shape=(knot_dim,)
        # clamp offsets into [0,1]
        raw_offset = np.clip(raw_knot, 0.0, 1.0)
        # add them to your *uniform* internal knots
        internal = np.sort(self.base_internal + raw_offset)

        # rebuild full open knot vector
        kv = ([0.0] * (self.degree+1)
            + internal.tolist()
            + [1.0] * (self.degree+1))

        return ctrl_pts, weights, kv


    def step(self, action):
        # ——— optionally disable learning of some params ———
        if not self.learn_ctrl:
            action[: self.ctrl_dim] = 0
        if not self.learn_weight:
            action[self.ctrl_dim : self.ctrl_dim + self.weight_dim] = 0
        if not self.learn_knot:
            action[-self.knot_dim :] = 0

        # 1) Apply delta to full state (ctrl pts + weights + knot offsets)
        self.state += self.step_size * action

        # 2) Unpack into control points, weights, and knot vector
        ctrl_pts, weights, kv = self._unpack_state()

        # 3) Build NURBS with all three sets of learnable parameters
        spline = sp.NURBS(
            degrees=[self.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
            weights=weights.tolist()
        )

        # 4) Sample & compute geometry exactly as before
        coords  = spline.evaluate(self.ts.reshape(-1,1))
        polygon = Polygon(zip(coords[:,0], coords[:,1]))

        # 5) Prepare verts for rendering
        scaled = coords / np.max(np.abs(coords)) * 100 + 300
        self.verts = list(zip(scaled[:,0], scaled[:,1]))

        # 6) Compute reward + done
        done   = (polygon.area == 0) or (self.num_step >= self.max_num_step)
        dist   = self._distance(spline, self.target_spline)
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
        """Start on the J shape, with control points, weights, and zero knot offsets."""
        self.num_step = 0

        # 1) control‐point portion = the big circle (flattened)
        ctrl    = self.initial_ctrl.copy()            # shape=(ctrl_dim,)

        # 2) weight portion = your preset J‐weights
        weights = self.initial_weights.copy()         # shape=(weight_dim,)

        # 3) knot‐offsets portion = zeros
        zeros   = np.zeros(self.knot_dim, dtype=np.float32)

        # → full state vector
        self.state = np.concatenate([ctrl, weights, zeros])

        # Precompute rendering verts exactly as before, but now using our new unpack:
        ctrl_pts, weights, kv = self._unpack_state()
        spline   = sp.NURBS(degrees=[self.degree],
                            knot_vectors=[kv],
                            control_points=ctrl_pts,
                            weights=weights.tolist())
        coords   = spline.evaluate(self.ts.reshape(-1,1))
        scaled   = coords/np.max(np.abs(coords))*100 + 300
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
        t = np.linspace(0,1,samples).reshape(-1,1) # Sample 100 values in [0, 1]
        pa = a.evaluate(t) # Get 100 (x, y) points from spline a
        pb = b.evaluate(t) # Get 100 (x, y) points from spline b
        return np.mean(np.linalg.norm(pa - pb, axis=1)) # Average L2 distance between corresponding points