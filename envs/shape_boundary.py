import math
from typing import Optional
import numpy as np
from shapely.geometry import Polygon
from gymnasium import spaces
import pygame
from pygame import gfxdraw
# BBO: A base environment implementing basic discounting and reward calculation
from envs.bbo import BBO
import splinepy as sp
from scipy.spatial import Delaunay
from envs.spline_utils import segments_intersect, _segment_distance, ccw

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
             ctrl_state_dim=18, 
             max_num_step=20, 
             render_mode='human', 
             degree=2, 
             n_internal_knots=14, 
             train_ctrl=True, 
             train_knot=True, 
             alpha_ctrl: float=0.2,
             alpha_knot: float=0.6,
             alpha_vel: float=0.3,
             alpha_energy: float=0.05,
             alpha_repulsion: float=0.1,
             repulse_k: float=7.0,
             repulse_epsilon: float=1e-3,
             lambda_decay: float = 3.0,
             repulse_r_max: float = 1.5,
             repulse_k_att: float = 1.0,
             # ——— spring energy ———
             alpha_spring: float = 0.1,
             k_spring: float = 1.0,
             # ——— edge‐length constraints ———
             min_edge_length: float = 0.05,
             max_edge_length: float = 2.0,
             lambda_edge_short: float = 10.0,
             lambda_edge_long: float = 5.0,
             alpha_edge: float = 0.2,
             # ——— intersection/barrier penalties ———
             alpha_intersect: float = 1.0,
             lambda_intersect: float = 100.0,
             d_min: float = 0.05,
             alpha_barrier: float = 0.5
            ):
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
        self.learn_knot   = train_knot
        
        # Reward-term weights
        self.alpha_ctrl       = alpha_ctrl       # rq1: geometry (distance → target)
        self.alpha_knot       = alpha_knot       # rq2: knot-vector alignment
        self.alpha_repulsion  = alpha_repulsion  # rq4: pairwise repulsion energy

        self.alpha_vel        = alpha_vel        # velocity penalty
        self.alpha_energy     = alpha_energy     # action-energy penalty

        self.repulse_k = repulse_k
        self.repulse_epsilon = repulse_epsilon
        
        self.sigma = 0.8 # desired “rest” distance
        self.lambda_decay = lambda_decay  # decay factor for repulsion energy
        
        self.repulse_r_max = repulse_r_max  # max distance for attraction
        self.repulse_k_att = repulse_k_att  # spring constant for attraction
        
        self.alpha_intersect   = 1.0    # overall weight for intersection penalty
        self.lambda_intersect  = 100.0  # barrier strength
        
        self.d_min = 0.05
        self.alpha_barrier = 0.5
        
        # spring weight & stiffness
        self.alpha_spring = alpha_spring
        self.k_spring     = k_spring
        
        # edge‐length constraints
        self.min_edge_length  = min_edge_length
        self.max_edge_length  = max_edge_length
        self.lambda_edge_short = lambda_edge_short
        self.lambda_edge_long  = lambda_edge_long
        self.alpha_edge       = alpha_edge

        # intersection / log‐barrier penalties
        self.alpha_intersect   = alpha_intersect
        self.lambda_intersect  = lambda_intersect
        self.d_min             = d_min
        self.alpha_barrier     = alpha_barrier
        
        # maximum allowed edge-length ratio between any two adjacent control points
        self.lambda_edge_ratio = 2.0   # λ = max_edge / min_edge ≤ 2
        
        # Edge‐based constraint monitoring
        # no edge may collapse below this length
        self.min_edge_length = 0.05
        # no edge may stretch beyond this length
        self.max_edge_length = 2.0

        # how strongly to penalize collapsed edges (< min)
        self.lambda_edge_short = 10.0  
        # how strongly to penalize over-stretched edges (> max)
        self.lambda_edge_long  = 5.0   

        # weight of the combined edge‐constraint penalty in your reward
        self.alpha_edge = 0.2
        
        # Initialize reward tracking
        self.last_rewards = {
            'ctrl':       0.0,
            'knot':       0.0,
            'repulsion':  0.0,
            'intersection': 0.0,
            'shape_quality': 0.0,
            'spring':     0.0,
            'total':      0.0
        }

        # spline sampling parameters
        self.ts   = np.linspace(0, 1, 80)
        self.verts = None
        
        # Build a *uniform open* knot vector for 36 control points
        # For 36 control points with degree 2, we need 36 + 2 + 1 = 39 knots
        # Format: [0,0,0, internal_knots, 1,1,1] for degree 2
        num_internal = 18 - self.degree - 1  # This should give us the right number of internal knots
        internal_knots = [i/(num_internal+1) for i in range(1, num_internal+1)]
        kv = [0.0] * (self.degree + 1) + internal_knots + [1.0] * (self.degree + 1)
        self.base_kv = kv

        # Update knot_dim to match the number of internal knots we can actually vary
        self.knot_dim = len(internal_knots)
        self.state_dim = self.ctrl_dim + self.knot_dim
        
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                    (self.state_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0,
                                    (self.state_dim,), dtype=np.float32)


        # store the *uniform* internal knots so offsets are added to these
        self.base_internal = np.array(internal_knots)

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
        
        # Define 5-pointed star shape (target shape)
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
        
        def densify_evenly(pts, target_count):
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
            
            return np.array(new_points[:target_count])
        
        # Create target spline with same 36-point structure
        # Build knot vector for 36 control points
        target_num_internal = self.knot_dim
        target_internal_knots = [i/(target_num_internal+1) for i in range(1, target_num_internal+1)]
        target_kv = [0.0] * (self.degree + 1) + target_internal_knots + [1.0] * (self.degree + 1)

        
        # For 36 control points, use n_subdivide = 2
        self.n_subdivide = 2

        # Densify to get 36 evenly spaced points
        J_pts = densify_evenly(J_pts, 18)
        E_pts = densify_evenly(E_pts, 18)

        # Set dimensions based on densified points
        self.num_coef   = len(J_pts)  # Should be 36
        self.ctrl_dim   = self.num_coef * 2  # 72
        self.state_dim  = self.ctrl_dim + self.knot_dim

        # Create target spline from the densified star shape
        self.target_spline = sp.BSpline(
            degrees=[self.degree],
            knot_vectors=[target_kv],
            control_points=E_pts,        
        )

        self.initial_ctrl = J_pts.flatten()

        # redefine your spaces to match
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (self.state_dim,), dtype=np.float32)
        
        # Repulsive spring step
        # Compute which control points to repel (non-adjacent in the Delaunay mesh)
        init_ctrl_pts = J_pts # shape=(num_coef, 2)
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
        
        # after self.edge_pairs = list(edges)
        init_pts = init_ctrl_pts  # your np.array of initial control‐points
        self.rest_lengths = [
            np.linalg.norm(init_pts[i] - init_pts[j])
            for i, j in self.edge_pairs
        ]


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

        # 3) raw knot offsets
        k_slice = slice(self.ctrl_dim, None)
        raw_knot = self.state[k_slice] # shape=(knot_dim,)
        # clamp offsets into [0,1]
        raw_offset = np.clip(raw_knot, 0.0, 1.0)
        # add them to your *uniform* internal knots
        # ensure base_internal + offset stays within [0,1]
        combined   = self.base_internal + raw_offset
        combined   = np.clip(combined, 0.0, 1.0)
        internal   = np.sort(combined)

        # rebuild full open knot vector
        kv = ([0.0] * (self.degree+1)
            + internal.tolist()
            + [1.0] * (self.degree+1))

        return ctrl_pts, kv
    
    def _compute_ctrl_reward(self, dist: float) -> float:
        # amplify geometry term
        return -2.0 * dist
    
    
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
    
    def _compute_enhanced_repulsion_energy(self, ctrl_pts):
        """
        Repulsion energy for any two control points or point-to-edge
        that come closer than self.d_min.
        """
        energy = 0.0
        n_pts = ctrl_pts.shape[0]

        # --- vertex-to-vertex repulsion ---
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = np.linalg.norm(ctrl_pts[i] - ctrl_pts[j])
                if d < self.d_min:
                    # 1/(d - d0 + eps)
                    energy += self.repulse_k / (d - self.d_min + self.repulse_epsilon)

        # --- vertex-to-edge repulsion ---
        # For each vertex i, check its distance to every non-adjacent segment (j → j+1)
        for i in range(n_pts):
            p = ctrl_pts[i]
            for j in range(n_pts):
                next_j = (j + 1) % n_pts
                # skip edges that share the vertex
                if i in (j, next_j):
                    continue
                p1, p2 = ctrl_pts[j], ctrl_pts[next_j]
                # compute point→segment distance; _segment_distance takes (p1,p2,p3,p4)
                # so we pass p twice to collapse one segment
                d_seg = _segment_distance(self, p1, p2, p, p)
                if d_seg < self.d_min:
                    energy += self.repulse_k / (d_seg - self.d_min + self.repulse_epsilon)

        return energy


    def _compute_intersection_penalty(self, ctrl_pts):
        """
        Enhanced intersection penalty with prevention and smooth gradients
        """
        penalty = 0.0
        n = ctrl_pts.shape[0]
        
        # 1. PREVENTION: Penalize segments getting too close (before intersection)
        near_intersection_penalty = 0.0
        
        # 2. PUNISHMENT: Heavily penalize actual intersections
        actual_intersection_penalty = 0.0
        
        for i in range(n):
            p1, p2 = ctrl_pts[i], ctrl_pts[(i+1) % n]
            
            for j in range(i+2, n):
                if i == 0 and j == n-1:  # Skip wrap-around adjacent pair
                    continue
                    
                p3, p4 = ctrl_pts[j], ctrl_pts[(j+1) % n]
                
                # Calculate minimum distance between segments
                min_dist = _segment_distance(self, p1, p2, p3, p4)

                
                # PREVENTION: Exponential penalty as segments get close
                if min_dist < 0.1:  # Danger zone
                    near_intersection_penalty += 500.0 * np.exp(-10.0 * min_dist)
                
                # PUNISHMENT: Massive penalty for actual intersections
                if segments_intersect(p1, p2, p3, p4):
                    actual_intersection_penalty += 10000.0  # Much larger base penalty
                    
                    # Additional penalty based on intersection severity
                    intersect_depth = self._calculate_intersection_depth(p1, p2, p3, p4)
                    actual_intersection_penalty += intersect_depth * 5000.0
        
        return near_intersection_penalty + actual_intersection_penalty  

    def _calculate_intersection_depth(self, p1, p2, p3, p4):
        """
        Calculate how severely two segments intersect
        """
        # Find intersection point using line intersection formula
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return 0.0  # Parallel lines
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        # Both t and u should be in [0,1] for intersection
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection point
            ix = x1 + t*(x2-x1)
            iy = y1 + t*(y2-y1)
            
            # Measure how far from endpoints (deeper intersection = worse)
            dist_from_ends = min(
                np.linalg.norm([ix-x1, iy-y1]),
                np.linalg.norm([ix-x2, iy-y2]),
                np.linalg.norm([ix-x3, iy-y3]),
                np.linalg.norm([ix-x4, iy-y4])
            )
            
            return 1.0 / (dist_from_ends + 1e-6)  # Deeper = higher penalty
        
        return 0.0

    def _compute_shape_quality_penalty(self, ctrl_pts):
        """
        Additional penalty for degenerate shapes
        """
        penalty = 0.0
        
        # Check for overlapping control points
        n = ctrl_pts.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(ctrl_pts[i] - ctrl_pts[j])
                if d < 1e-3:  # Nearly identical points
                    penalty += 1000.0 * (1e-3 - d)
        
        # Check for extreme aspect ratios
        x_range = np.ptp(ctrl_pts[:, 0])  # Peak-to-peak (max - min)
        y_range = np.ptp(ctrl_pts[:, 1])
        
        if min(x_range, y_range) > 0:
            aspect_ratio = max(x_range, y_range) / min(x_range, y_range)
            if aspect_ratio > 10.0:  # Very elongated shape
                penalty += 10.0 * (aspect_ratio - 10.0)
        
        return penalty
    
    def _compute_spring_energy(self, ctrl_pts: np.ndarray) -> float:
        """
        Sum ½·k·(d - d0)**2 over all adjacent edges.
        """
        energy = 0.0
        for idx, (i, j) in enumerate(self.edge_pairs):
            d  = np.linalg.norm(ctrl_pts[i] - ctrl_pts[j])
            d0 = self.rest_lengths[idx]
            # you can also make k dynamic, e.g. k = self.k_spring / d0
            k  = self.k_spring
            energy += 0.5 * k * (d - d0)**2
        return energy

    def _compute_edge_length_penalty(self, ctrl_pts: np.ndarray) -> float:
        """
        Penalize edges that are too short or too long.
        """
        penalty = 0.0
        n = ctrl_pts.shape[0]
        for i in range(n):
            j = (i + 1) % n
            d = np.linalg.norm(ctrl_pts[i] - ctrl_pts[j])
            if d < self.min_edge_length:
                # quadratic penalty for collapse
                penalty += self.lambda_edge_short * (self.min_edge_length - d)**2
            elif d > self.max_edge_length:
                # quadratic penalty for overstretch
                penalty += self.lambda_edge_long * (d - self.max_edge_length)**2
        return penalty

    def step(self, action):
        # --- unpack old ctrl_pts for velocity term ---
        old_ctrl, _ = self._unpack_state()
        self.prev_ctrl_pts = old_ctrl.copy()

        # optionally disable learning of some params
        if not self.learn_ctrl:
            action[:self.ctrl_dim] = 0
        if not self.learn_knot:
            action[-self.knot_dim :] = 0

        # 1) Apply delta to full state (ctrl pts + weights + knot offsets)
        self.state += self.step_size * action

        # 2) Unpack into control points, weights, and knot vector
        ctrl_pts, kv = self._unpack_state()
        
        # 3) Build NURBS with all three sets of learnable parameters
        spline = sp.BSpline(
            degrees=[self.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
        )

        # 4) Sample & compute geometry exactly as before
        coords  = spline.evaluate(self.ts.reshape(-1,1))
        polygon = Polygon(zip(coords[:,0], coords[:,1]))

        # 5) Prepare verts for rendering
        scaled = coords / np.max(np.abs(coords)) * 100 + 300
        self.verts = list(zip(scaled[:,0], scaled[:,1]))

        # 6) Compute individual sub‐rewards
        # 6a) Control‐point geometry reward
        dist = self._distance(spline, self.target_spline)
        reward_ctrl = self._compute_ctrl_reward(dist) if self.learn_ctrl else 0.0
        internal = np.array(kv[self.degree+1:-(self.degree+1)])
        reward_knot = self._compute_knot_reward(internal) if self.learn_knot else 0.0
        
        reward_vel = self._compute_velocity_reward(ctrl_pts) if self.learn_ctrl else 0.0
        reward_energy = self._compute_energy_penalty(action)
        
        # 2. Compute constraint penalties with enhanced methods
        rep_energy = self._compute_enhanced_repulsion_energy(ctrl_pts)
        intersection_penalty = self._compute_intersection_penalty(ctrl_pts)
        shape_penalty = self._compute_shape_quality_penalty(ctrl_pts)
        spring_energy      = self._compute_spring_energy(ctrl_pts)
        edge_penalty = self._compute_edge_length_penalty(ctrl_pts)
        
        # 3. Make constraints dominant early
        t_norm = self.num_step / float(self.max_num_step)
        
        # Geometry reward should be suppressed if constraints are violated
        constraint_violation = rep_energy + intersection_penalty + shape_penalty
        if constraint_violation > 1000:  # Significant violations
            geometry_suppression = 0.1  # Reduce geometry reward to 10%
        else:
            geometry_suppression = 1.0

        # 6f) Dynamic weighting via exponential decay
        decay = math.exp(-self.lambda_decay * t_norm)
        alpha_ctrl_dyn   = self.alpha_ctrl   *  decay
        alpha_knot_dyn   = self.alpha_knot   * (1 - decay) * 0.5

        # 6g) Combine everything with enhanced constraints
        total_reward = (
            alpha_ctrl_dyn * reward_ctrl * geometry_suppression +
            alpha_knot_dyn * reward_knot +
            self.alpha_vel * reward_vel +
            self.alpha_energy * reward_energy +
            -self.alpha_repulsion * rep_energy +
            -self.alpha_intersect * intersection_penalty +
            -self.alpha_barrier * shape_penalty +
            - self.alpha_spring     * spring_energy +
            - self.alpha_edge       * edge_penalty
        )

        # 7) Save for analysis/plotting - include new constraint info
        self.last_rewards = {
            'ctrl':         reward_ctrl,
            'knot':         reward_knot,
            'repulsion':    -rep_energy,
            'intersection': -intersection_penalty,
            'shape_quality': -shape_penalty,
            'vel':          reward_vel,
            'energy':       reward_energy,
            'spring':       -spring_energy,
            'edge':         -edge_penalty,
            'total':        total_reward
        }


        # 8) Termination check
        done = (polygon.area == 0) or (self.num_step >= self.max_num_step)
        self.num_step += 1

        # 9) Return new state, reward, done, truncated, info with enhanced info
        return self.state.copy(), total_reward, done, False, {
            "repulsion_energy": rep_energy,
            "intersection_penalty": intersection_penalty,
            "shape_penalty": shape_penalty,
            "spring_energy": spring_energy,
            "edge_penalty": edge_penalty,
            "constraint_violation": constraint_violation,
            "geometry_suppression": geometry_suppression
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
        """Start on the J shape, with control points and zero knot offsets."""

        self.num_step = 0

        # 1) control‐point portion = the big circle (flattened)
        ctrl   = self.initial_ctrl.copy()            # length = ctrl_dim

        # 3) knot‐offsets portion = zeros
        zeros   = np.zeros(self.knot_dim, dtype=np.float32)

        # → full state vector
        self.state = np.concatenate([ctrl, zeros])

        # Precompute rendering verts exactly as before, but now using our new unpack:
        ctrl_pts, kv = self._unpack_state()
        spline   = sp.BSpline(degrees=[self.degree],
                            knot_vectors=[kv],
                            control_points=ctrl_pts)
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