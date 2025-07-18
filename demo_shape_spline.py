# demo_shape_spline_enhanced.py
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import splinepy as sp
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from envs.shape_boundary import ShapeBoundary
import torch
from policy import Policy
from common_nets import Mlp
from utils import get_train_params, get_architectures

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
            'ctrl_reward': '#FF4500', # Orange red
            'knot_reward': '#4169E1',  # Blue violet
            'repulsion': '#8A2BE2'     # ADD THIS: Dark magenta for repulsion
        }
        
        # Store trajectory data
        self.trajectory_data = []
        
        # Store separate reward histories
        self.reward_history = []
        self.ctrl_reward_history = []
        self.knot_reward_history = []
        self.distance_history = []
        
        # Store knot histories
        self.knot_history = []
        
        # Store repulsion history
        self.repulsion_reward_history = []
        self.repulsion_energy_history = []
        self.control_point_distances = [] # For heatmap visualization
        
        # Store initial values for comparison
        self.initial_knots = None
        
    def create_frame(self, step, obs, reward, done=False):
        """Create a single visualization frame showing the current state of the spline deformation."""
        # Extract the flattened control point values from the observation vector
        ctrl_flat = obs[:self.env.ctrl_dim]
        
        # Reshape the flat array into a 2D (num_control_points × 2) array
        ctrl_pts = ctrl_flat.reshape(self.env.num_coef, 2)
        
        # Unpack the knot vector from the environment state (used to define the NURBS spline)
        _, kv = self.env._unpack_state()
        
        # Store initial values on first frame
        if self.initial_knots is None:
            self.initial_knots = np.array(kv)
        
        # Store current values for history
        self.knot_history.append(np.array(kv))
        
        # Rebuild the current BSpline spline using the control points and knot vector
        current_spline = sp.BSpline(
            degrees=[self.env.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
        )
        
        # Evaluate the spline at sampled time steps to get its current 2D shape
        curr_pts = current_spline.evaluate(self.env.ts.reshape(-1, 1))

        # combine control points + artificial “scaffolding” points
        all_verts = ctrl_pts.copy()
        delaunay = Delaunay(all_verts)
        triangles = delaunay.simplices

        
        # Compute and save the centroid (average position) of the current control points
        centroid = np.mean(ctrl_pts, axis=0)
        self.trajectory_data.append(centroid.copy())
        
        # Store reward histories (get individual rewards from environment)
        if hasattr(self.env, 'last_rewards'):
            self.ctrl_reward_history.append(self.env.last_rewards['ctrl'])
            self.knot_reward_history.append(self.env.last_rewards['knot'])
            self.reward_history.append(self.env.last_rewards['total'])
            
            self.repulsion_reward_history.append(self.env.last_rewards['repulsion'])
            self.reward_history.append(self.env.last_rewards['total'])
            
            # raw enhanced‐repulsion energy
            repulsion_energy = self.env._compute_enhanced_repulsion_energy(ctrl_pts)
            self.repulsion_energy_history.append(repulsion_energy)
            
            distances = self._compute_distance_matrix(ctrl_pts)
            self.control_point_distances.append(distances)
        else:
            # Fallback if environment doesn't have separate rewards
            self.reward_history.append(reward)
            self.ctrl_reward_history.append(reward)
            self.knot_reward_history.append(0.0)
            self.repulsion_reward_history.append(0.0)
            self.repulsion_energy_history.append(0.0)
        
        # Compute distance for distance history
        dist = self.env._distance(current_spline, self.env.target_spline)
        self.distance_history.append(dist)
        
        
        # Create a new figure with a custom background color and size
        fig = plt.figure(figsize=self.figsize, facecolor=self.colors['background'])
        # Create a 4-row by 4-column grid for subplots with spacing defined
        gs = fig.add_gridspec(3, 4, height_ratios=[3, 1, 1], width_ratios=[2, 2, 1, 1], 
                         hspace=0.4, wspace=0.3)
        
        # Plot 1: Main shape comparison (target vs. current spline)
        ax_main = fig.add_subplot(gs[0, :2]) # Span first two columns of top row
        self._plot_main_shapes(ax_main, curr_pts, ctrl_pts, all_verts, triangles, step, dist, done)
        
        # # Plot 2: Trajectory of the control point centroid
        # ax_traj = fig.add_subplot(gs[0, 1])  # Right column of top row
        # self._plot_trajectory(ax_traj)
        
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
        
        # # Plot 7: Knot evolution
        # ax_knots = fig.add_subplot(gs[2, :2])
        # self._plot_knot_evolution(ax_knots, kv, step)
        
        # Plot 8: Knot history
        ax_knot_hist = fig.add_subplot(gs[2, 0])
        self._plot_knot_history(ax_knot_hist)
        
        # Plot 10: Repulsion energy history
        ax_repulsion_hist = fig.add_subplot(gs[2, 2:])
        self._plot_repulsion_history(ax_repulsion_hist)
        
        # # Plot 13: Control point distance heatmap
        # ax_distance_heatmap = fig.add_subplot(gs[4, 1])
        # self._plot_distance_heatmap(ax_distance_heatmap, ctrl_pts)
        
        # # Plot 14: Repulsion force visualization
        # ax_repulsion_forces = fig.add_subplot(gs[4, 2:])
        # self._plot_repulsion_forces(ax_repulsion_forces, ctrl_pts)

        # Automatically adjust layout so elements don’t overlap
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_argb()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        frame = img[:, :, 1:4]  # ARGB → RGB
        plt.close(fig)
        
        return frame\
            
    def _plot_reward_contributions(self, ax):
        """Plot current reward contributions as pie chart"""
        if not hasattr(self.env, 'last_rewards') or len(self.ctrl_reward_history) == 0:
            ax.text(0.5, 0.5, 'No reward\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Reward Contributions', fontsize=10, weight='bold')
            return
        
        # Get current reward contributions (absolute values for pie chart)
        current_rewards = self.env.last_rewards
        ctrl_contrib = abs(current_rewards['ctrl']) * self.env.alpha_ctrl
        knot_contrib = abs(current_rewards['knot']) * self.env.alpha_knot
        repulsion_contrib = abs(current_rewards['repulsion']) * self.env.alpha_repulsion
        
        # Only show non-zero contributions
        labels = []
        sizes = []
        colors = []
        
        if ctrl_contrib > 1e-6:
            labels.append(f'Control\n({current_rewards["ctrl"]:.3f})')
            sizes.append(ctrl_contrib)
            colors.append(self.colors['ctrl_reward'])
        
        if knot_contrib > 1e-6 and self.env.learn_knot:
            labels.append(f'Knots\n({current_rewards["knot"]:.3f})')
            sizes.append(knot_contrib)
            colors.append(self.colors['knot_reward'])
            
        if repulsion_contrib > 1e-6:
            labels.append(f'Repulsion\n({current_rewards["repulsion"]:.3f})')
            sizes.append(repulsion_contrib)
            colors.append(self.colors['repulsion'])
        
        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                  startangle=90, textprops={'fontsize': 8})
        else:
            ax.text(0.5, 0.5, 'All rewards\nare zero', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Current Reward Contributions', fontsize=10, weight='bold')
    
    def _plot_main_shapes(self, ax, curr_pts, ctrl_pts, all_verts, triangles, step, dist, done):
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
        
        # overlay the triangulation mesh
        tri = Triangulation(all_verts[:,0], all_verts[:,1], triangles)
        ax.triplot(tri, color='gray', linewidth=0.5, alpha=0.6)
        
        
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
        
        # Draw arrows showing how control points moved from previous step
        if len(self.trajectory_data) > 1:  # Skip this on first step
            for i, ctrl_pt in enumerate(ctrl_pts):
                if i < len(self.prev_ctrl_pts):  # Make sure previous point exists
                    dx = ctrl_pt[0] - self.prev_ctrl_pts[i, 0]  # Change in X
                    dy = ctrl_pt[1] - self.prev_ctrl_pts[i, 1]  # Change in Y
                    # Only draw arrow if movement is big enough to see
                    if np.sqrt(dx**2 + dy**2) > 0.01:
                        ax.arrow(
                            self.prev_ctrl_pts[i, 0], self.prev_ctrl_pts[i, 1],  # Start
                            dx, dy,                                              # Direction
                            head_width=0.05, head_length=0.05,                   # Arrowhead size
                            fc=self.colors['trajectory'], ec=self.colors['trajectory'],  # Fill and edge color
                            alpha=0.7                                            # Slight transparency
                        )
        # draw triangulation on top of the white background
        tri = Triangulation(all_verts[:,0], all_verts[:,1], triangles)
        ax.triplot(tri, color='gray', linewidth=0.5, alpha=0.6)

        
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
    
    def _plot_trajectory(self, ax):
        """Plot the trajectory of control point centroid"""
        if len(self.trajectory_data) > 1:
            traj = np.array(self.trajectory_data)
            
            # Create a gradient line for trajectory
            points = traj.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis', alpha=0.8)
            lc.set_array(np.arange(len(segments)))
            ax.add_collection(lc)
            
            # Mark start and current position
            ax.scatter(traj[0, 0], traj[0, 1], c='red', s=100, 
                      marker='o', label='Start', zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=100, 
                      marker='*', label='Current', zorder=5)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Centroid Trajectory', fontsize=10, weight='bold')
        ax.legend(fontsize=8)
        
    def _plot_individual_rewards(self, ax):
        """Plot individual reward components"""
        if len(self.ctrl_reward_history) < 1:
            return
            
        steps = range(len(self.ctrl_reward_history))
        
        # Plot each reward component
        ax.plot(steps, self.ctrl_reward_history, 
               color=self.colors['ctrl_reward'], linewidth=2, 
               label='Control Points', marker='o', markersize=3)
        
        if self.env.learn_knot:
            ax.plot(steps, self.knot_reward_history, 
                   color=self.colors['knot_reward'], linewidth=2, 
                   label='Knots', marker='^', markersize=3)
            
            
        ax.plot(steps, self.repulsion_reward_history,
               color=self.colors['repulsion'], linewidth=2,
           label='Repulsion', marker='d', markersize=3)
        
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
        
    def _plot_knot_evolution(self, ax, current_kv, step):
        """Show current knot positions vs initial/target"""
        if self.initial_knots is None:
            return
            
        # Extract internal knots (exclude repeated boundary knots)
        degree = self.env.degree
        current_internal = np.array(current_kv[degree+1:-degree-1])
        initial_internal = np.array(self.initial_knots[degree+1:-degree-1])
        target_internal = np.array(self.env.target_spline.knot_vectors[0][degree+1:-degree-1])
        
        # Plot knot positions as vertical lines
        y_pos = np.arange(len(current_internal))
        
        # Initial knots (light gray)
        ax.barh(y_pos - 0.2, initial_internal, height=0.3, 
                color='lightgray', alpha=0.7, label='Initial')
        
        # Target knots (green)
        ax.barh(y_pos, target_internal, height=0.3, 
                color=self.colors['target'], alpha=0.7, label='Target')
        
        # Current knots (orange)
        ax.barh(y_pos + 0.2, current_internal, height=0.3, 
                color=self.colors['knots'], alpha=0.8, label='Current')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'K{i+1}' for i in range(len(current_internal))])
        ax.set_xlabel('Knot Value')
        ax.set_title('Knot Positions', fontsize=10, weight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    
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
    
    def _compute_distance_matrix(self, ctrl_pts):
        """Compute distance matrix between all control points"""
        n_pts = len(ctrl_pts)
        distances = np.zeros((n_pts, n_pts))
        for i in range(n_pts):
            for j in range(n_pts):
                if i != j:
                    distances[i, j] = np.linalg.norm(ctrl_pts[i] - ctrl_pts[j])
        return distances
    
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

    def _plot_distance_heatmap(self, ax, ctrl_pts):
        """Plot heatmap of distances between control points"""
        distances = self._compute_distance_matrix(ctrl_pts)
        
        im = ax.imshow(distances, cmap='RdYlBu_r', aspect='equal')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Mark non-adjacent pairs that contribute to repulsion
        for i, j in self.env.non_adjacent_pairs:
            # Draw rectangle around non-adjacent pairs
            rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_title('Control Point Distances', fontsize=10, weight='bold')
        ax.set_xlabel('Control Point Index')
        ax.set_ylabel('Control Point Index')

    def _plot_repulsion_forces(self, ax, ctrl_pts):
        """Visualize repulsion forces between non-adjacent control points"""
        # Plot control points
        ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], 
                c='blue', s=60, zorder=5, edgecolors='white', linewidth=1.5)
        
        # Draw repulsion forces as lines between non-adjacent pairs
        for i, j in self.env.non_adjacent_pairs:
            pt_i, pt_j = ctrl_pts[i], ctrl_pts[j]
            distance = np.linalg.norm(pt_i - pt_j)
            
            # Color and thickness based on repulsion strength
            force_strength = self.env.repulse_k / (distance + self.env.repulse_epsilon)
            normalized_strength = min(force_strength / 10.0, 1.0)  # Normalize for visualization
            
            # Draw line with color indicating force strength
            ax.plot([pt_i[0], pt_j[0]], [pt_i[1], pt_j[1]], 
                color='red', alpha=0.3 + 0.7 * normalized_strength,
                linewidth=0.5 + 2 * normalized_strength)
            
            # Add distance text for closest pairs
            if distance < 0.3:  # Only show for very close points
                mid_point = (pt_i + pt_j) / 2
                ax.text(mid_point[0], mid_point[1], f'{distance:.2f}', 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        
        ax.set_aspect('equal')
        ax.set_title('Repulsion Forces\n(Red lines = repulsive pairs)', fontsize=10, weight='bold')
        ax.grid(True, alpha=0.3)


def create_enhanced_visualization():
    """Create the enhanced spline deformation visualization"""
    # make sure device exists before loading the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the shape deformation environment with enhanced constraint settings
    env = ShapeBoundary(
        naive=False,              # Use shaped reward, not just raw negative value
        step_size=3e-2,           # Smaller steps for better stability
        ctrl_state_dim=16,        # 18 control points × 2D = 36-dimensional state
        max_num_step=200,         # More steps for convergence
        render_mode="rgb_array",  # Output images (used for making frames)
        degree=3,                 # Use cubic splines (degree 3)
        n_internal_knots=0,      # Number of internal knots controls spline flexibility
        train_ctrl=True,          # Train control points
        train_knot=False,          # Train knot positions
        
        # ADJUSTED REWARD WEIGHTS
        alpha_ctrl=0.6,           # Stronger pull toward target geometry
        alpha_knot=0.15,          # Less emphasis on knot alignment
        alpha_repulsion=0.1,      # Will be handled by enhanced constraints
        alpha_vel=0.08,           # Less velocity penalty
        alpha_energy=0.03,        # Less action penalty
        
        # ENHANCED CONSTRAINT PARAMETERS
        repulse_k=8.0,            # Stronger base repulsion
        repulse_epsilon=1e-4,     # Better numerical stability
        repulse_r_max=1.8,        # Larger attraction zone
        repulse_k_att=1.2,        # Stronger attraction
        lambda_decay=2.5,         # Moderate transition timing
        
        # BARRIER AND CONSTRAINT PARAMETERS
        # These are new parameters you may need to add to your __init__ method:
        min_edge_length=0.03,     # Minimum allowed edge length
        max_edge_length=2.5,      # Maximum allowed edge length
        lambda_edge_short=15.0,   # Penalty for collapsed edges
        lambda_edge_long=8.0,     # Penalty for over-stretched edges
        alpha_edge=0.15,          # Weight for edge constraint penalty
        
        # Intersection penalties
        alpha_intersect=0.8,      # Weight for intersection penalty
        lambda_intersect=150.0,   # Barrier strength for intersections
        
        # Barrier parameters
        d_min=0.04,               # Minimum distance for log barrier
        alpha_barrier=0.6,        # Weight for barrier penalty
    )
    
    # DPO policy setup
    ckpt = torch.load("models/shape_boundary_DPO_first_order.pth", map_location=device)
    
    # build the network with exactly the same layers that the checkpoint expects
    in_dim, out_dim = env.state_dim, env.state_dim      # both should be 16 here
    layer_dims     = get_architectures("shape_boundary",
                                      zero_order=False)  # e.g. [32, 64, 32]
    rate, _, step_size, _, _, _ = get_train_params("shape_boundary")

    main_net = Mlp(
        input_dim=in_dim,
        output_dim=out_dim,
        layer_dims=layer_dims
    ).to(device)

    policy = Policy(
        zero_order=False,
        main_net=main_net,
        rate=rate,
        step_size=step_size
    )
    policy.main_net.load_state_dict(ckpt)
    policy.main_net.eval()
    prev_action = np.zeros(env.state_dim, dtype=np.float32)
    
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
    print(f"Knot component weight: {env.alpha_knot}")
    print(f"Repulsion component weight: {env.alpha_repulsion}")
    
    # Run the simulation for a fixed number of steps
    for step in range(1, env.max_num_step + 1):
        # ─ use trained DPO policy to pick the action ─
        action = policy.get_action(obs, prev_action)
        prev_action = action.copy()
        
        # Apply the action to the environment and get the next observation, reward, and done flag
        obs, reward, done, _, info = env.step(action)
        
        # Print reward breakdown every 10 steps
        if step % 10 == 0 and hasattr(env, 'last_rewards'):
            crev = env.last_rewards['repulsion']
            cene = info['repulsion_energy']
            print(f"Step {step}: Total={env.last_rewards['total']:.3f}, "
                f"Ctrl={env.last_rewards['ctrl']:.3f}, "
                f"Knot={env.last_rewards['knot']:.3f}, "
                f"RepulsionReward={crev:.3f}, "
                f"RepulsionEnergy={cene:.3f}")
        
        # Create a new frame showing the updated spline and performance
        frame = viz.create_frame(step, obs, reward, done)
        frames.append(frame)
        
        if done:
            # Add a few extra frames at the end to show final result
            for _ in range(3):
                frames.append(frame)
            break
    
    # Save as GIF
    imageio.mimsave("letters_spline_deformation.gif", frames, fps=69, loop=0)
    print("Created letters_spline_deformation.gif")
    
    return frames

if __name__ == "__main__":
    create_enhanced_visualization()
    