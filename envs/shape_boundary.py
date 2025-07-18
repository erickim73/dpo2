from typing import Optional
import numpy as np
from shapely.geometry import Polygon
from gymnasium import spaces
import pygame
from pygame import gfxdraw
from envs.bbo import BBO
import splinepy as sp
from scipy.spatial import Delaunay
import math

# A large numeric constant used to penalize degenerate shapes
MAX_ACT = 1e4

class ShapeBoundary(BBO):
    metadata = {
        # Supported rendering modes: interactive human window or RGB array for video
        "render_modes": ["human", "rgb_array"],
        # Framers per second when rendering
        "render_fps": 15,
    }

    def __init__(self, naive=False, 
                 step_size=1e-2, 
                 ctrl_state_dim=16, 
                 max_num_step=20, 
                 render_mode='human', 
                 degree=2, 
                 n_internal_knots=0, 
                 train_ctrl=True, 
                 train_weight=False, 
                 train_knot=False, 
                 alpha_ctrl: float=0.2,
                 alpha_weight: float=0.2,
                 alpha_knot: float=0.6,
                 alpha_repulsion: float=0.05, # weight for repulsive spring penalty
                 repulse_k: float=0.5, # spring constant k
                 repulse_epsilon: float=1e-3, # small term to avoid division by zero
                 repulse_r_max: float = 1.5, # above this distance apply attraction
                 repulse_k_att: float = 1.0, # spring constant for attraction
                 alpha_tri_quality: float = 0.1,    # Weight for triangle quality regularizer
                 min_tri_radius: float = 0.1,       # Min acceptable circumradius
                 max_tri_radius: float = 1.0,       # Max acceptable circumradius
                 lambda_tri_small: float = 5.0,     # Penalty for small radii
                 lambda_tri_large: float = 2.0,     # Penalty for large radii
                ):
        # Initialize the base BBO environment
        #  - naive: if True, use simple reward = -val. else use pmp-shaped reward
        #  - step_size: scaling factor how much each action perturbs the step
        #  - max_num_step: maximum number of steps before episode terminates
        super(ShapeBoundary, self).__init__(naive, step_size, max_num_step)

        # spline + knot control parameters
        self.ctrl_dim        = ctrl_state_dim                   # e.g. 36 control points (2D×18)
        self.num_coef        = ctrl_state_dim // 2
        self.weight_dim = 0 if not train_weight else self.num_coef
        self.knot_dim   = 0 if not train_knot   else n_internal_knots
        
        # total state = [ctrl_pts (2D×num_coef) + weights + knot‐offsets]
        self.state_dim  = self.ctrl_dim + self.weight_dim + self.knot_dim

        # now redefine your spaces to match the new dimension:
        self.observation_space = spaces.Box(-4, 4, (self.state_dim,),  dtype=np.float32)
        self.action_space      = spaces.Box(-1, 1, (self.state_dim,), dtype=np.float32)

        self.degree          = degree
        self.max_num_step    = max_num_step
        self.step_size       = step_size
        
        self.learn_ctrl   = train_ctrl
        self.learn_weight = train_weight
        self.learn_knot   = train_knot
        
        # ——— Reward-term weights ———
        self.alpha_ctrl       = alpha_ctrl       # rq1: geometry (distance → target)
        self.alpha_knot       = alpha_knot       # rq2: knot-vector alignment
        self.alpha_weight     = alpha_weight     # rq3: weight-vector alignment
        self.alpha_repulsion  = alpha_repulsion  # rq4: pairwise repulsion energy

        self.repulse_k = repulse_k
        self.repulse_epsilon = repulse_epsilon
        
        self.sigma = 0.8 # desired “rest” distance
        
        self.repulse_r_max = repulse_r_max  # max distance for attraction
        self.repulse_k_att = repulse_k_att  # spring constant for attraction
        
        # weight of the combined edge‐constraint penalty in your reward
        self.alpha_edge = 0.2
        
        # --- Triangle Quality Regularizer ---
        self.alpha_tri_quality = alpha_tri_quality
        self.min_tri_radius = min_tri_radius
        self.max_tri_radius = max_tri_radius
        self.lambda_tri_small = lambda_tri_small
        self.lambda_tri_large = lambda_tri_large

        
        # Initialize reward tracking
        self.last_rewards = {
            'ctrl':      0.0,
            'weight':    0.0,
            'knot':      0.0,
            'repulsion': 0.0,
            'tri_quality': 0.0,
            'total':     0.0
        }

        # spline sampling parameters
        self.ts   = np.linspace(0, 1, 80)
        self.verts = None
        
        # Define initial & target circles
        # Build a *uniform open* knot vector using exactly n_internal_knots
        internal_knots = [i / (n_internal_knots + 1) for i in range(1, n_internal_knots + 1)]
        kv = [0.0] * (self.degree + 1) + internal_knots + [1.0] * (self.degree + 1)
        self.base_kv = kv
        # store the *uniform* internal knots so offsets are added to these
        self.base_internal = np.array(
            kv[self.degree+1 : - (self.degree+1)]
        )
        
        # 1. Letter J (starting shape) → Letter E (target shape)
        start_pts_template = np.array([
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

        target_pts_template = np.array([
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

        start_pts = self._densify_evenly(start_pts_template, self.num_coef)
        target_pts = self._densify_evenly(target_pts_template, self.num_coef)
        
        # Use weights to emphasize key structural points
        start_weights = np.ones(self.num_coef)
        target_weights = np.ones(self.num_coef)
        

        self.initial_weights = np.array(start_weights, dtype=np.float32)
        
        # Create the target spline (E shape)
        self.target_spline = sp.NURBS(
            degrees=[self.degree],
            knot_vectors=[kv],
            control_points=target_pts,
            weights=target_weights
        )
        
        self.start_weights = start_weights
        # as a numpy array for easy math later
        self.initial_weights = np.array(self.start_weights, dtype=np.float32)

        # flatten for initial state (J shape)
        self.initial_ctrl = start_pts.flatten()
        
        # Repulsive spring step
        # Compute which control points to repel (non-adjacent in the Delaunay mesh)
        init_ctrl_pts = start_pts # shape=(num_coef, 2)
        delaunay = Delaunay(init_ctrl_pts)
        # Collect all edges from each triangle
        edges = {
            tuple(sorted(e))
            for tri in delaunay.simplices
            for e in [(tri[0],tri[1]), (tri[1],tri[2]), (tri[0],tri[2])]
        }
        n_pts = init_ctrl_pts.shape[0]
        # All i<j pairs that are not in the edges
        self.non_adjacent_pairs = [
            (i,j)
            for i in range(n_pts) for j in range(i+1, n_pts)
            if (i,j) not in edges
        ]
        # include the *adjacent* Delaunay edges too
        self.edge_pairs = list(edges)
        # final list of all springs
        self.pair_list   = self.edge_pairs + self.non_adjacent_pairs

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 600
        self.screen = None # Pygame window or surface
        self.clock = None # Pygame clock for FPS control
        self.isopen = True # Tracks if the window is open
        
        # Spline specific paramters
        self.n_internal_knots = n_internal_knots  # Number of internal knots in the spline
        
    def _densify_evenly(self, pts, target_count):
            """
            Create evenly spaced points along the perimeter of a closed polygon
            """
            # Calculate perimeter distances
            N = len(pts)
            edge_lengths = []
            for i in range(N):
                p1 = pts[i]
                p2 = pts[(i+1) % N]
                edge_lengths.append(np.linalg.norm(p2 - p1))
            
            total_perimeter = sum(edge_lengths)
            target_spacing = total_perimeter / target_count
            
            # Generate evenly spaced points
            new_points = []
            current_distance = 0.0
            next_target = 0.0
            
            for i in range(N):
                p1 = pts[i]
                p2 = pts[(i+1) % N]
                edge_vec = p2 - p1
                edge_len = edge_lengths[i]
                
                # Add points along this edge
                while next_target <= current_distance + edge_len and len(new_points) < target_count:
                    if next_target <= current_distance:
                        # Point is at the start of this edge
                        new_points.append(p1.copy())
                    else:
                        # Point is partway along this edge
                        t = (next_target - current_distance) / edge_len
                        new_point = p1 + t * edge_vec
                        new_points.append(new_point)
                    next_target += target_spacing
                
                current_distance += edge_len
            
            # Ensure we have exactly target_count points
            while len(new_points) < target_count:
                new_points.append(pts[-1].copy())
                
            new_points = np.asarray(new_points) 
                
            # 1) make both rings clockwise
            if Polygon(new_points).area > 0:        # Shapely gives +area for CCW polygons
                new_points = new_points[::-1]

            # 2) move the left-most point to index 0 – that becomes the common seam
            start = np.argmin(new_points[:, 0])
            new_points = np.roll(new_points, -start, axis=0)
            
            return np.array(new_points[:target_count])
        
        
    def set_state(self, state):
        """Manually sets the environment's state from the outside."""
        self.state = state.copy()
        # We also update the previous control points to prevent a large velocity penalty
        # from the jump to the new state.
        if hasattr(self, 'state'):
            self.prev_ctrl_pts, _, _ = self._unpack_state()
    
    def _unpack_state(self):
        # 1) control points
        flat 	= self.state[:self.ctrl_dim]
        ctrl_pts = flat.reshape(self.num_coef, 2)

        # 2) weights
        w_slice = slice(self.ctrl_dim, self.ctrl_dim + self.weight_dim)
        weights = self.state[w_slice]      # may be length-0
        if self.weight_dim == 0:
            weights = np.ones(self.num_coef, dtype=np.float32)  # fabricate

        # 3) knot vector
        # If knots are part of the state, calculate the new knot vector.
        # Otherwise, just return the original base knot vector.
        if self.knot_dim > 0:
            k_slice = slice(self.ctrl_dim + self.weight_dim, None)
            raw_knot = self.state[k_slice]

            # clamp each knot-offset
            max_offsets = 1.0 - self.base_internal
            raw_offset = np.clip(raw_knot, 0.0, max_offsets)

            # build and sort your internal knots
            internal = np.sort(self.base_internal + raw_offset)

            # rebuild full open knot vector
            kv = (
                [0.0] * (self.degree + 1)
                + internal.tolist()
                + [1.0] * (self.degree + 1)
            )
        else:
            # No knots in the state vector, so use the unmodified base knot vector
            kv = self.base_kv

        return ctrl_pts, weights, kv
    
    def _compute_ctrl_reward(self, dist: float) -> float:
        # amplify geometry term
        return -2.0 * dist
    
    def _compute_weight_reward(self, weights: np.ndarray) -> float:
        # encourage weights -> target weights (mean absolute error)
        target = np.asarray(self.target_spline.weights, dtype=np.float32)
        return -float(np.mean(np.abs(weights - target)))
    
    def _compute_knot_reward(self, internal: np.ndarray) -> float:
        # encourage internal knots -> target (mean absolute error ×2)
        targ = np.array(
            self.target_spline.knot_vectors[0][self.degree+1:-(self.degree+1)]
        )
        return -5.0 * float(np.mean(np.abs(internal - targ)))
    
    def _compute_velocity_reward(self, ctrl_pts: np.ndarray) -> float:
        # penalize large instantaneous velocity of control points
        if not hasattr(self, 'prev_ctrl_pts'):
            return 0.0
        v = ctrl_pts - self.prev_ctrl_pts
        speed = np.linalg.norm(v, axis=1).mean()
        return -float(speed)

    def _compute_energy_penalty(self, action: np.ndarray) -> float:
        # penalize squared‐magnitude of the action vector (approx. work/energy)
        return -float(np.sum(action**2))
    
    def _compute_pairwise_energy(self, ctrl_pts: np.ndarray, k_rep: float, k_att: float) -> float:
        """
        For each pair:
         - if d < sigma: ½·k_rep·(σ - d)²  (repulsion)
         - elif d > repulse_r_max:  ½·k_att·(d - σ)²  (weak attraction)
         - else: 0
        """
        energy = 0.0
        for i, j in self.pair_list:
            d = np.linalg.norm(ctrl_pts[i] - ctrl_pts[j]) + self.repulse_epsilon
            if d < self.sigma:
                δ = self.sigma - d
                energy += 0.5 * k_rep * (δ * δ) 
            elif d > self.repulse_r_max:
                Δ = d - self.sigma
                energy += 0.5 * k_att * (Δ * Δ) 
        return energy
    
    def _compute_triangle_quality_penalty(self, ctrl_pts: np.ndarray) -> float:
        """
        Computes a penalty based on the quality of triangles in the Delaunay mesh.
        Penalizes triangles with circumradii that are too small or too large.
        """
        # Ensure there are enough points for at least one triangle
        if len(ctrl_pts) < 3:
            return 0.0

        try:
            # 1. Perform Delaunay triangulation
            delaunay = Delaunay(ctrl_pts)
        except Exception: # Catches QHullError for collinear/degenerate points
            return MAX_ACT # Return a large penalty for degenerate configurations

        total_penalty = 0.0
        
        # Iterate over each triangle (simplex) in the mesh
        for tri_indices in delaunay.simplices:
            p1, p2, p3 = ctrl_pts[tri_indices]

            # 2. Calculate side lengths of the triangle
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)

            # Avoid division by zero for degenerate triangles (zero area)
            if a * b * c < 1e-9:
                continue

            # 3. Calculate area using Heron's formula for robustness
            s = (a + b + c) / 2.0
            area_squared = s * (s - a) * (s - b) * (s - c)
            if area_squared <= 1e-9: # Check for non-positive area from floating point errors
                continue
            area = np.sqrt(area_squared)

            # 4. Calculate the circumradius: R = abc / 4A
            circum_radius = (a * b * c) / (4.0 * area)

            # 5. Calculate penalty if radius is outside the acceptable range
            small_r_penalty = max(0.0, self.min_tri_radius - circum_radius)
            large_r_penalty = max(0.0, circum_radius - self.max_tri_radius)
            
            total_penalty += self.lambda_tri_small * small_r_penalty + self.lambda_tri_large * large_r_penalty
        
        return total_penalty

    def step(self, action):
        # unpack old ctrl_pts for velocity term
        old_ctrl, _, _ = self._unpack_state()
        self.prev_ctrl_pts = old_ctrl.copy()

        # optionally disable learning of some params
        if not self.learn_ctrl:
            action[:self.ctrl_dim] = 0
        if not self.learn_weight:
            action[self.ctrl_dim : self.ctrl_dim + self.weight_dim] = 0
        if not self.learn_knot:
            action[-self.knot_dim :] = 0

        # 1) Apply delta to full state (ctrl pts + weights + knot offsets)
        self.state += self.step_size * action

        # 2) Unpack into control points, weights, and knot vector
        ctrl_pts, weights, kv = self._unpack_state()

        #  dynamic repulsion/attraction strength
        t_norm = self.num_step / float(self.max_num_step)
        # e.g. ramp up repulsion over time
        k_dyn = self.repulse_k * (0.5 + 0.5 * t_norm)  
        k_att_dyn = self.repulse_k_att * (1.0 - t_norm)   # optional: attraction fades out
        rep_energy = self._compute_pairwise_energy(ctrl_pts, k_dyn, k_att_dyn) \
                     if self.learn_ctrl else 0.0
        # rq4: convert raw repulsion‐energy into a reward term
        reward_rep = -self.alpha_repulsion * rep_energy
        
        # Compute Triangle Quality Penalty
        tri_penalty = self._compute_triangle_quality_penalty(ctrl_pts)
        reward_tri_quality = -self.alpha_tri_quality * tri_penalty

        # 3) Build NURBS with all three sets of learnable parameters
        w_for_spline = weights if self.weight_dim else np.ones(self.num_coef)
        spline = sp.NURBS(
            degrees=[self.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
            weights=w_for_spline.tolist()
        )

        # 4) Sample & compute geometry exactly as before
        coords  = spline.evaluate(self.ts.reshape(-1,1))
        polygon = Polygon(zip(coords[:,0], coords[:,1]))

        # 5) Prepare verts for rendering
        scaled = coords / np.max(np.abs(coords)) * 100 + 300
        self.verts = list(zip(scaled[:,0], scaled[:,1]))

        # 6) Compute individual sub‐rewards
        # 6a) Control‐point geometry reward
        coords = spline.evaluate(self.ts.reshape(-1, 1))
        (
            para_coords,
            phys_coords,
            phys_diff,
            dist_arr,
            conv_norm,
            deriv1,
            deriv2,
        ) = self.target_spline.proximities(coords, return_verbose=True)
        mean_dist = float(dist_arr.mean())
        reward_ctrl = self._compute_ctrl_reward(mean_dist) if self.learn_ctrl else 0.0

        # 6b) Weight reward
        reward_weight = self._compute_weight_reward(weights) if self.learn_weight else 0.0

        # 6c) Knot reward
        internal = np.array(kv[self.degree+1:-(self.degree+1)])
        reward_knot = self._compute_knot_reward(internal) if self.learn_knot else 0.0

        # 6f) Simplified Reward Calculation
        # The total reward is based on the distance to target and a repulsion penalty.
        total_reward = reward_ctrl + reward_rep + reward_tri_quality

        # 7) Save for analysis/plotting
        # We still save all components so the plots work, but only ctrl_reward affects the agent's goal.
        self.last_rewards = {
            'ctrl':      reward_ctrl,
            'knot':      reward_knot,
            'weight':    reward_weight,
            'repulsion': reward_rep,
            'tri_quality': reward_tri_quality,
            'total':     total_reward
        }

        # 8) Termination check
        done = (polygon.area == 0) or (self.num_step >= self.max_num_step)
        self.num_step += 1

        # 9) Return new state, reward, done, truncated, info
        return self.state.copy(), total_reward, done, False, {
            "repulsion_energy": rep_energy
        }
    
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

        parts = [ctrl]
        if self.weight_dim:
            parts.append(self.initial_weights.copy())
        if self.knot_dim:
            parts.append(np.zeros(self.knot_dim, dtype=np.float32))
        self.state = np.concatenate(parts)

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
            
    