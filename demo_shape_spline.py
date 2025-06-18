# demo_shape_spline_enhanced.py
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import splinepy as sp
from matplotlib.animation import FuncAnimation

from envs.shape_boundary import ShapeBoundary

class EnhancedSplineVisualizer:
    def __init__(self, env, figsize=(12, 8)):
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
            'text': '#2F4F4F'         # Dark slate gray
        }
        
        # Store trajectory data
        self.trajectory_data = []
        self.reward_history = []
        self.distance_history = []
        
    def create_frame(self, step, obs, reward, done=False):
        """Create a single visualization frame showing the current state of the spline deformation."""
        # Extract the flattened control point values from the observation vector
        ctrl_flat = obs[:self.env.ctrl_dim]
        
        # Reshape the flat array into a 2D (num_control_points × 2) array
        ctrl_pts = ctrl_flat.reshape(self.env.num_coef, 2)
        
        # Unpack the knot vector from the environment state (used to define the NURBS spline)
        _, kv = self.env._unpack_state()
        
        # Rebuild the current NURBS spline using the control points and knot vector
        current_spline = sp.NURBS(
            degrees=[self.env.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
            weights=[1.0] * self.env.num_coef # Equal weights for all control points
        )
        # Evaluate the spline at sampled time steps to get its current 2D shape
        curr_pts = current_spline.evaluate(self.env.ts.reshape(-1, 1))
        
        # Compute and save the centroid (average position) of the current control points
        centroid = np.mean(ctrl_pts, axis=0)
        self.trajectory_data.append(centroid.copy())
        
        # Store the reward value for the reward history plot
        self.reward_history.append(reward)
        
        # Compute and store the distance to the target (used for distance plot)
        dist = -reward  # The reward is negative distance, so distance = -reward
        self.distance_history.append(dist)
        
        
        # Create a new figure with a custom background color and size
        fig = plt.figure(figsize=self.figsize, facecolor=self.colors['background'])
        # Create a 2-row by 3-column grid for subplots with spacing defined
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Plot 1: Main shape comparison (target vs. current spline)
        ax_main = fig.add_subplot(gs[0, :2]) # Span first two columns of top row
        self._plot_main_shapes(ax_main, curr_pts, ctrl_pts, step, dist, done)
        
        # Plot 2: Trajectory of the control point centroid
        ax_traj = fig.add_subplot(gs[0, 2])  # Right column of top row
        self._plot_trajectory(ax_traj)
        
        # Plot 3: Reward history
        ax_reward = fig.add_subplot(gs[1, 0])  # Bottom-left
        self._plot_reward_history(ax_reward)

        # Plot 4: Distance-to-target history
        ax_dist = fig.add_subplot(gs[1, 1])  # Bottom-center
        self._plot_distance_history(ax_dist)

        # Plot 5: Circular progress indicator
        ax_progress = fig.add_subplot(gs[1, 2])  # Bottom-right
        self._plot_progress_indicator(ax_progress, step, done)
        
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
    
    def _plot_main_shapes(self, ax, curr_pts, ctrl_pts, step, dist, done):
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
        
        # Status text with better formatting
        status = "CONVERGED!" if done and dist < 0.1 else "LEARNING..."
        status_color = self.colors['target'] if done and dist < 0.1 else self.colors['current']
        
        ax.text(
            1.05, 0.8,  # Position to the right of the plot area
            f"Step: {step:2d}/{self.env.max_num_step}\n"
            f"Distance: {dist:.4f}\n"
            f"Status: {status}",
            transform=ax.transAxes,  # Position in axes-relative units
            va='top', ha='left',
            bbox=dict(
                boxstyle="round,pad=0.5",        # Rounded box
                facecolor='white',              # White background
                edgecolor=status_color,         # Edge color depends on status
                alpha=0.9                       # Semi-transparent
            ),
            fontsize=10,
            color=self.colors['text'],  # Dark gray text
            weight='bold'
        )
        
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


def create_enhanced_visualization():
    """Create the enhanced spline deformation visualization"""
    # Create the shape deformation environment with specific settings
    env = ShapeBoundary(
        naive=False,              # Use shaped reward, not just raw negative value
        step_size=5e-2,           # How much an action perturbs the spline
        ctrl_state_dim=36,        # 18 control points × 2D = 36-dimensional state
        max_num_step=120,          # Total number of steps to run the simulation
        render_mode="rgb_array",  # Output images (used for making frames)
        degree=3,                 # Use cubic splines (degree 3)
        n_internal_knots=4        # Number of internal knots controls spline flexibility
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
    
    # Run the simulation for a fixed number of steps
    for step in range(1, env.max_num_step + 1):
        # Generate a smart action that tries to move the spline closer to the target
        action = generate_smart_action(env, obs)
        
        # Apply the action to the environment and get the next observation, reward, and done flag
        obs, reward, done, _, _ = env.step(action)
        # Create a new frame showing the updated spline and performance
        frame = viz.create_frame(step, obs, reward, done)
        # Add the frame to the animation
        frames.append(frame)
        
        if done:
            # Add a few extra frames at the end to show final result
            for _ in range(3):
                frames.append(frame)
            break
    
    # Save as GIF
    imageio.mimsave("letters_spline_deformation.gif", frames, fps=6, loop=0)
    print("Created letters_spline_deformation.gif")
    
    return frames

def generate_smart_action(env, obs):
    """Generate a somewhat intelligent action that moves toward the target"""
    # Get the current control points from the observation
    ctrl_flat = obs[:env.ctrl_dim]  # Extract the flat control point vector from the observation
    ctrl_pts = ctrl_flat.reshape(env.num_coef, 2)  # Reshape into a 2D array (num_control_points × 2D)
    
    # pull target control points directly from the env
    E_ctrl = env.target_spline.control_points  # shape (8,2)
    diff     = E_ctrl - ctrl_pts               # move toward E
    action_ctrl = diff.flatten() * 0.5

    # Add noise to make the motion less deterministic
    noise = np.random.normal(0, 0.1, size=action_ctrl.shape)  # Gaussian noise with std=0.1
    action_ctrl += noise  # Add noise to the action
    
    # Add action values for knot vector updates if knot_dim > 0
    if env.knot_dim > 0:
        # Generate random small changes for knot values
        knot_action = np.random.normal(0, 0.05, size=env.knot_dim)

        # Combine control point action and knot vector action into one array
        action = np.concatenate([action_ctrl, knot_action])
    else:
        # Only control points are updated
        action = action_ctrl
    
    # Clip to action space
    return np.clip(action, env.action_space.low, env.action_space.high)

if __name__ == "__main__":
    create_enhanced_visualization()