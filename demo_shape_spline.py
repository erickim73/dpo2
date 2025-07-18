# demo_shape_spline_enhanced.py
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import splinepy as sp
from scipy.spatial import Delaunay
from envs.shape_boundary import ShapeBoundary

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

class EnhancedSplineVisualizer:
    def __init__(self, env, figsize=(18, 12)):
        self.env = env
        self.figsize = figsize
        # Evaluate the target spline at sampled time points ts to get target shape coordinates
        # env.ts is a 1D array of parameter values in [0,1], reshape(-1, 1) makes it 2D (required by splinepy)
        self.target_pts = env.target_spline.evaluate(env.ts.reshape(-1, 1))
        
        # Color palette
        self.colors = {
            'target': '#2E8B57',      # Sea green
            'current': '#4169E1',     # Royal blue
            'control': '#FF6347',     # Tomato
            'trajectory': '#9370DB',  # Medium purple
            'background': '#F8F8FF',  # Ghost white
            'text': '#2F4F4F',        # Dark slate gray
            'knots': '#FF8C00',       # Dark orange
            'weights': '#DC143C',     # Crimson
            'ctrl_reward': '#FF4500', # Orange red
            'weight_reward': '#32CD32', # Lime green 
            'knot_reward': '#4169E1',  # Blue violet
            'repulsion': '#8A2BE2',      # Dark magenta for repulsion
           'tri_quality': '#00CED1'     # Dark turquoise for triangle quality
        }
        
        # Store separate reward histories
        self.reward_history = []
        self.ctrl_reward_history = []
        self.weight_reward_history = []
        self.knot_reward_history = []
        self.tri_quality_reward_history = [] 
        self.distance_history = []
        
        # Store knot and weight histories
        self.knot_history = []
        self.weight_history = []
        
        # Store repulsion history
        self.repulsion_reward_history = []
        self.repulsion_energy_history = []
        
        # Store initial values for comparison
        self.initial_knots = None
        self.initial_weights = None
        
    def create_frame(self, step, obs, reward, done=False):
        """Create a single visualization frame showing the current state of the spline deformation."""
        # Extract the flattened control point values from the observation vector
        ctrl_flat = obs[:self.env.ctrl_dim]
        
        # Reshape the flat array into a 2D (num_control_points × 2) array
        ctrl_pts = ctrl_flat.reshape(self.env.num_coef, 2)
        
        # Unpack the knot vector from the environment state (used to define the NURBS spline)
        _, weights, kv = self.env._unpack_state()
        
        # Store initial values on first frame
        if self.initial_knots is None:
            self.initial_knots = np.array(kv)
            self.initial_weights = weights.copy()
        
        # Store current values for history
        self.knot_history.append(np.array(kv))
        self.weight_history.append(weights.copy())
        
        # Rebuild the current NURBS spline using the control points and knot vector
        current_spline = sp.NURBS(
            degrees=[self.env.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
            weights=weights
        )
        
        # Evaluate the spline at sampled time steps to get its current 2D shape
        curr_pts = current_spline.evaluate(self.env.ts.reshape(-1, 1))
        
        # Store reward histories (get individual rewards from environment)
        if hasattr(self.env, 'last_rewards'):
            self.ctrl_reward_history.append(self.env.last_rewards['ctrl'])
            self.weight_reward_history.append(self.env.last_rewards['weight'])
            self.knot_reward_history.append(self.env.last_rewards['knot'])
            self.repulsion_reward_history.append(self.env.last_rewards['repulsion'])
            self.tri_quality_reward_history.append(self.env.last_rewards['tri_quality'])
            self.reward_history.append(self.env.last_rewards['total'])
            
            # raw pairwise energy (repulsion & attraction)
            repulsion_energy = self.env._compute_pairwise_energy(
                ctrl_pts,
                self.env.repulse_k,
                self.env.repulse_k_att
            )
            self.repulsion_energy_history.append(repulsion_energy)
            
        else:
            # Fallback if environment doesn't have separate rewards
            self.reward_history.append(reward)
            self.ctrl_reward_history.append(reward)
            self.weight_reward_history.append(0.0)
            self.knot_reward_history.append(0.0)
            self.repulsion_reward_history.append(0.0)
            self.repulsion_energy_history.append(0.0)
        
        # Compute distance for distance history
        mean_dist = 0.0  # Initialize with a default value
        if 'ctrl' in self.env.last_rewards and self.env.last_rewards['ctrl'] is not None:
            mean_dist = -self.env.last_rewards['ctrl'] / 2.0
        self.distance_history.append(mean_dist)
        
        
        # Create a new figure with a custom background color and size
        fig = plt.figure(figsize=self.figsize, facecolor=self.colors['background'])
        # Create a 4-row by 4-column grid for subplots with spacing defined
        gs = fig.add_gridspec(3, 4, height_ratios=[3, 1, 1], width_ratios=[2, 2, 1, 1], 
                         hspace=0.4, wspace=0.3)
        
        # Plot 1: Main shape comparison (target vs. current spline)
        ax_main = fig.add_subplot(gs[0, :2]) # Span first two columns of top row
        self._plot_main_shapes(ax_main, curr_pts, ctrl_pts)
        
        # Plot 3: Circular progress indicator
        ax_progress = fig.add_subplot(gs[0, 2]) # Top far right
        self._plot_progress_indicator(ax_progress, step, done)
        
        # Plot 4: Reward contribution pie chart
        ax_pie = fig.add_subplot(gs[0, 3])
        self._plot_reward_contributions(ax_pie)
        
        # Plot 5: Combined reward history
        ax_reward = fig.add_subplot(gs[1, 0]) #Bottom left
        self._plot_reward_history(ax_reward)

        # Plot 6: Distance-to-target history
        ax_dist = fig.add_subplot(gs[1, 1])  # Bottom-center
        self._plot_distance_history(ax_dist)
        
        # Plot 7: Individual reward histories
        ax_individual = fig.add_subplot(gs[1, 2:])
        self._plot_individual_rewards(ax_individual)
        
        # Plot 8: Knot history
        ax_knot_hist = fig.add_subplot(gs[2, 0])
        self._plot_knot_history(ax_knot_hist)
        
        # Plot 9: Weight history
        ax_weight_hist = fig.add_subplot(gs[2, 1])
        self._plot_weight_history(ax_weight_hist)
        
        # Plot 10: Repulsion energy history
        ax_repulsion_hist = fig.add_subplot(gs[2, 2:])
        self._plot_repulsion_history(ax_repulsion_hist)
        
        # Automatically adjust layout so elements don’t overlap
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_argb()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        frame = img[:, :, 1:4]  # ARGB → RGB
        plt.close(fig)
        
        return frame
            
    def _plot_reward_contributions(self, ax):
        """Plot current reward contributions as pie chart"""
        if not hasattr(self.env, 'last_rewards') or len(self.ctrl_reward_history) == 0:
            ax.text(0.5, 0.5, 'No reward\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Reward Contributions', fontsize=10, weight='bold')
            return
        
        # Get current reward contributions (absolute values for pie chart)
        current_rewards = self.env.last_rewards
        ctrl_contrib = abs(current_rewards.get('ctrl', 0.0))
        weight_contrib = abs(current_rewards.get('weight', 0.0))
        knot_contrib = abs(current_rewards.get('knot', 0.0))
        repulsion_contrib = abs(current_rewards.get('repulsion', 0.0))
        tri_quality_contrib = abs(current_rewards.get('tri_quality', 0.0))
        
        # Only show non-zero contributions
        labels = []
        sizes = []
        colors = []
        
        if ctrl_contrib > 1e-6:
            labels.append(f'Control\n({current_rewards["ctrl"]:.3f})')
            sizes.append(ctrl_contrib)
            colors.append(self.colors['ctrl_reward'])
        
        if weight_contrib > 1e-6 and self.env.learn_weight:
            labels.append(f'Weights\n({current_rewards["weight"]:.3f})')
            sizes.append(weight_contrib)
            colors.append(self.colors['weight_reward'])
        
        if knot_contrib > 1e-6 and self.env.learn_knot:
            labels.append(f'Knots\n({current_rewards["knot"]:.3f})')
            sizes.append(knot_contrib)
            colors.append(self.colors['knot_reward'])
            
        if repulsion_contrib > 1e-6:
            labels.append(f'Repulsion\n({current_rewards["repulsion"]:.3f})')
            sizes.append(repulsion_contrib)
            colors.append(self.colors['repulsion'])
        
        if tri_quality_contrib > 1e-6:
            labels.append(f'Tri Quality\n({current_rewards["tri_quality"]:.3f})')
            sizes.append(tri_quality_contrib)
            colors.append(self.colors['tri_quality'])
        
        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                  startangle=90, textprops={'fontsize': 8})
        else:
            ax.text(0.5, 0.5, 'All rewards\nare zero', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Current Reward Contributions', fontsize=10, weight='bold')
    
    def _plot_main_shapes(self, ax, curr_pts, ctrl_pts):
        """Plot the current shape, target shape, control points, and deformation vectors."""
    
        # Draw the target shape as a filled polygon
        target_poly = patches.Polygon(
            self.target_pts,         # Precomputed sample points from the target spline
            closed=True,             # Make the polygon closed
            facecolor=self.colors['target'],  # Fill color (e.g., sea green)
            alpha=0.3,               # Semi-transparent fill
            edgecolor=self.colors['target'],  # Edge color same as fill
            linewidth=2,             # Border thickness
            label='Target'           # Label for legend
        )

        # Draw the current shape (deformed spline) as a filled polygon
        current_poly = patches.Polygon(
            curr_pts,                 # Sampled points from the current spline
            closed=True,
            facecolor=self.colors['current'],  # Fill color (royal blue)
            alpha=0.2,
            edgecolor=self.colors['current'],
            linewidth=2,
            label='Current'
        )
        
        # Add both shapes to the plot
        ax.add_patch(target_poly)
        ax.add_patch(current_poly)
        
        # Draw the control points of the current spline
        ax.scatter(
            ctrl_pts[:, 0], ctrl_pts[:, 1],  # X and Y coordinates of control points
            c=self.colors['control'],        # Point color (e.g., tomato red)
            s=60,                            # Marker size
            zorder=5,                        # Draw on top of polygons
            edgecolors='white',              # White outline around each point
            linewidth=1.5,                   # Outline thickness
            label='Control Points'
        )
        
        target_ctrl_pts = self.env.target_spline.control_points
        ax.scatter(
            target_ctrl_pts[:, 0], target_ctrl_pts[:, 1], # Target X and Y
            c=self.colors['target'],  # Use the target's color (sea green)
            s=80,                     # Make them slightly larger
            marker='x',               # Use an 'x' marker to distinguish them
            zorder=4,                 # Place them just behind the current points
            label='Target Points'
        )
        
        self._plot_colored_delaunay_triangulation(ax, ctrl_pts)
        
        # Connect the control points with dashed lines to show their order
        for i in range(len(ctrl_pts)):
            next_i = (i + 1) % len(ctrl_pts)  # Wrap around to connect last to first
            ax.plot(
                [ctrl_pts[i, 0], ctrl_pts[next_i, 0]],  # X coordinates
                [ctrl_pts[i, 1], ctrl_pts[next_i, 1]],  # Y coordinates
                '--',                                  # Dashed line
                color=self.colors['control'],
                alpha=0.5,                              # Slightly transparent
                linewidth=1
            )
        
        # Store current control points for use in next frame's movement arrows
        self.prev_ctrl_pts = ctrl_pts.copy()
        
        # Styling
        ax.set_aspect('equal', adjustable='box')  # Keep x/y scale equal
        ax.grid(True, alpha=0.3)                  # Light grid lines
        ax.set_facecolor(self.colors['background'])  # Light background color
        
        # Dynamic zoom based on shape bounds
        all_pts = np.vstack([self.target_pts, curr_pts])  # Combine all shape points
        margin = 0.2  # Add padding around shapes
        x_range = [all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin]
        y_range = [all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin]
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # Add a legend and title
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
        ax.set_title('Spline Shape Optimization', fontsize=14, weight='bold', 
                    color=self.colors['text'])
        
    def _plot_colored_delaunay_triangulation(self, ax, ctrl_pts):
        """Draws the Delaunay triangulation, shading only the 'bad' triangles."""
        if len(ctrl_pts) < 3:
            return

        try:
            tri = Delaunay(ctrl_pts)
        except Exception:
            return # Cannot triangulate, so skip drawing

        # 1. Draw all triangle edges with a light, neutral color to show the mesh
        ax.triplot(ctrl_pts[:, 0], ctrl_pts[:, 1], tri.simplices,
                   color='gray', lw=0.75, alpha=0.5, zorder=3)

        min_r = self.env.min_tri_radius
        max_r = self.env.max_tri_radius
        cmap = plt.get_cmap('coolwarm') # blue = too small, red = too large

        # 2. Iterate through triangles and shade only the bad ones
        for simplex in tri.simplices:
            p1, p2, p3 = ctrl_pts[simplex]

            # Circumradius calculation
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)

            # Use Heron's formula for area and check for degenerate triangles
            s = (a + b + c) / 2.0
            area_squared = s * (s - a) * (s - b) * (s - c)
            if area_squared <= 1e-9 or a * b * c < 1e-9:
                continue
            area = np.sqrt(area_squared)
            circum_radius = (a * b * c) / (4.0 * area)
            
            # Determine if triangle is "bad" and what color it should be
            color_to_use = None
            if circum_radius < min_r:
                # Triangle is too small (blue)
                color_val = 0.5 * (circum_radius / min_r)
                color_to_use = cmap(color_val)
            elif circum_radius > max_r:
                # Triangle is too large (red)
                color_val = 0.5 + 0.5 * min(1.0, (circum_radius - max_r) / max_r)
                color_to_use = cmap(color_val)
            
            # If the triangle was determined to be bad, shade it
            if color_to_use is not None:
                triangle_patch = patches.Polygon(
                    [p1, p2, p3],
                    closed=True,
                    facecolor=color_to_use,
                    alpha=0.6,    # Semi-transparent fill
                    edgecolor=None, # No border on the patch
                    zorder=4      # Draw on top of the gray mesh
                )
                ax.add_patch(triangle_patch)
    
    def _plot_individual_rewards(self, ax):
        """Plot individual reward components"""
        if len(self.ctrl_reward_history) < 1:
            return
            
        steps = range(len(self.ctrl_reward_history))
        
        # Plot each reward component
        ax.plot(steps, self.ctrl_reward_history, 
               color=self.colors['ctrl_reward'], linewidth=2, 
               label='Control Points', marker='o', markersize=3)
        
        if self.env.learn_weight:
            ax.plot(steps, self.weight_reward_history, 
                   color=self.colors['weight_reward'], linewidth=2, 
                   label='Weights', marker='s', markersize=3)
        
        if self.env.learn_knot:
            ax.plot(steps, self.knot_reward_history, 
                   color=self.colors['knot_reward'], linewidth=2, 
                   label='Knots', marker='^', markersize=3)
            
            
        ax.plot(steps, self.repulsion_reward_history,
            color=self.colors['repulsion'], linewidth=2,
            label='Repulsion', marker='d', markersize=3)
        
        clipped_tri_quality = np.clip(self.tri_quality_reward_history, -200, None)
        ax.plot(steps, clipped_tri_quality,
            color=self.colors['tri_quality'], linewidth=2,
            label='Tri Quality', marker='x', markersize=3)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title('Individual Reward Components', fontsize=10, weight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.legend(fontsize=8)
    
    def _plot_reward_history(self, ax):
        """Plot reward history"""
        if self.reward_history:
            steps = range(len(self.reward_history))
            ax.plot(steps, self.reward_history, color=self.colors['current'], 
                   linewidth=2, marker='o', markersize=4)
            ax.fill_between(steps, self.reward_history, alpha=0.3, 
                           color=self.colors['current'])
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Reward History', fontsize=10, weight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
    
    def _plot_distance_history(self, ax):
        """Plot distance history"""
        if self.distance_history:
            steps = range(len(self.distance_history))
            ax.plot(steps, self.distance_history, color=self.colors['target'], 
                   linewidth=2, marker='s', markersize=4)
            ax.fill_between(steps, self.distance_history, alpha=0.3, 
                           color=self.colors['target'])
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Distance to Target', fontsize=10, weight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance')
        ax.set_yscale('log')  # Log scale for better visualization
    
    def _plot_progress_indicator(self, ax, step, done):
        """Plot circular progress indicator"""
        progress = step / self.env.max_num_step
        
        # Draw progress circle
        circle = patches.Circle((0.5, 0.5), 0.4, fill=False, 
                               edgecolor=self.colors['text'], linewidth=8)
        ax.add_patch(circle)
        
        # Draw progress arc
        if progress > 0:
            theta = np.linspace(0, 2 * np.pi * progress, 100)
            x_arc = 0.5 + 0.4 * np.cos(theta - np.pi/2)
            y_arc = 0.5 + 0.4 * np.sin(theta - np.pi/2)
            ax.plot(x_arc, y_arc, color=self.colors['current'], linewidth=8)
        
        # Center text
        color = self.colors['target'] if done else self.colors['text']
        ax.text(0.5, 0.5, f"{step}\n/{self.env.max_num_step}", 
               ha='center', va='center', fontsize=12, weight='bold', color=color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Progress', fontsize=10, weight='bold')
        
    def _plot_knot_history(self, ax):
        """Plot how knots change over time"""
        if len(self.knot_history) < 2:
            return
            
        degree = self.env.degree
        steps = range(len(self.knot_history))
        
        # Plot evolution of each internal knot
        for i in range(len(self.knot_history[0]) - 2*(degree+1)):
            knot_values = [kv[degree+1+i] for kv in self.knot_history]
            ax.plot(steps, knot_values, marker='o', markersize=2, 
                   label=f'Knot {i+1}' if i < 3 else '', alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Knot Evolution', fontsize=10, weight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Knot Value')
        if len(self.knot_history[0]) - 2*(degree+1) <= 3:
            ax.legend(fontsize=8)
    
    def _plot_weight_history(self, ax):
        """Plot how weights change over time"""
        if len(self.weight_history) < 2:
            return
            
        steps = range(len(self.weight_history))
        weight_array = np.array(self.weight_history)
        
        # Plot evolution of first few weights (to avoid clutter)
        for i in range(min(5, weight_array.shape[1])):
            ax.plot(steps, weight_array[:, i], marker='o', markersize=2, 
                   label=f'Weight {i+1}', alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Weight Evolution', fontsize=10, weight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Weight Value')
        ax.legend(fontsize=8)
        
    def _plot_repulsion_history(self, ax):
        """Plot repulsion energy and reward over time"""
        if len(self.repulsion_energy_history) < 1:
            ax.text(0.5, 0.5, 'No repulsion\ndata available', 
                ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Repulsion History', fontsize=10, weight='bold')
            return
        
        steps = range(len(self.repulsion_energy_history))
        
        # Plot repulsion energy
        ax2 = ax.twinx()
        line1 = ax.plot(steps, self.repulsion_reward_history,
                    color=self.colors['repulsion'], linewidth=2, label='Repulsion Reward')
        line2 = ax2.plot(steps, self.repulsion_energy_history, 
                        color='red', linewidth=2, linestyle='--', label='Repulsion Energy')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Repulsion Reward', color='purple')
        ax2.set_ylabel('Repulsion Energy', color='red')
        ax.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Repulsion History', fontsize=10, weight='bold')


def create_enhanced_visualization():
    """Create the enhanced spline deformation visualization"""
    # Create the shape deformation environment with specific settings
    env = ShapeBoundary(
        step_size      = 0.05,      # safe first guess
        ctrl_state_dim = 48,
        max_num_step   = 120,
        render_mode    = "rgb_array",
        degree         = 3,
        n_internal_knots = 20,

        # — learning switches —
        train_ctrl     = True,
        train_weight   = False,
        train_knot     = False,

        # — reward weights —
        alpha_ctrl     = 1.0,   # keep
        alpha_weight   = 0.0,   
        alpha_knot     = 0.0,   
        alpha_repulsion= 0.05,   
        
        alpha_tri_quality = 0.2,   # Set a non-zero weight to activate it
        min_tri_radius    = 0.1,
        max_tri_radius    = 1.0
    )
    
    # Initialize visualizer
    viz = EnhancedSplineVisualizer(env)
    viz.prev_ctrl_pts = np.zeros((env.num_coef, 2))  # Initialize previous control points (used to draw movement arrows)
    
    # Reset the environment to get the initial observation (spline control points, etc.)
    obs, _ = env.reset()
    frames = []
    
    # Create the first frame (step 0) with reward 0.0
    initial_frame = viz.create_frame(0, obs, 0.0)
    frames.append(initial_frame)
    
    print("Starting simulation with separate reward components:")
    print(f"Control point weight: {env.alpha_ctrl}")
    print(f"Weight component weight: {env.alpha_weight}")
    print(f"Knot component weight: {env.alpha_knot}")
    print(f"Repulsion component weight: {env.alpha_repulsion}")
    
    def align_ring(src, tgt):
        n = src.shape[0]
        best_shift, best_flip, best_cost = 0, False, np.inf
        for flipped in (False, True):
            r = tgt[::-1] if flipped else tgt
            for s in range(n):
                diff = src - np.roll(r, s, axis=0)
                cost = np.linalg.norm(diff, axis=1).sum()
                if cost < best_cost:
                    best_cost, best_shift, best_flip = cost, s, flipped
        r = tgt[::-1] if best_flip else tgt
        return np.roll(r, best_shift, axis=0)

    for step in range(1, env.max_num_step + 1):
        # 1. Unpack the CURRENT observation to get the control points.
        ctrl_pts = obs[:env.ctrl_dim].reshape(env.num_coef, 2)
        target_ctrl_pts = env.target_spline.control_points
        
        # 2. Align the target to the current shape.
        aligned_target = align_ring(ctrl_pts, target_ctrl_pts)
        
        # 3. Calculate the direction and the action vector.
        direction_vectors = aligned_target - ctrl_pts
        action_control = (direction_vectors / env.step_size).flatten()

        max_abs = np.abs(action_control).max()
        if max_abs > 1.0:
            action_control /= max_abs
        
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[:env.ctrl_dim] = action_control

        # 4. Manually calculate the next state ('obs')
        #    This is the key change to force the update.
        obs = obs + env.step_size * action
        env.set_state(obs)

        # 5. Call step with a zero action ONLY to get the reward for the new state.
        #    The observation returned here is ignored.
        _, reward, done, _, info = env.step(np.zeros_like(action))

        # 6. Create the visualization frame with our manually updated state.
        frame = viz.create_frame(step, obs, reward, done)
        frames.append(frame)

        # 7. Check for convergence and update the 'done' flag.
        mean_distance = np.linalg.norm(direction_vectors, axis=1).mean()
        if mean_distance < 1e-4:
            print(f"Converged at step {step} with mean distance: {mean_distance:.5e}")
            done = True

        if done:
            frames.extend([frame] * 15)
            break
        
    # Save as GIF
    imageio.mimsave("spline_deformation_separate_rewards7.gif", frames, fps=5, loop=0)
    print("Created spline_deformation_separate_rewards7.gif")
    
    return frames

if __name__ == "__main__":
    create_enhanced_visualization()
    