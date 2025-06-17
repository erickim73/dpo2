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
        """Create a single frame of the visualization"""
        # Rebuild current spline
        ctrl_flat = obs[:self.env.ctrl_dim]
        ctrl_pts = ctrl_flat.reshape(self.env.num_coef, 2)
        _, kv = self.env._unpack_state()
        current_spline = sp.NURBS(
            degrees=[self.env.degree],
            knot_vectors=[kv],
            control_points=ctrl_pts,
            weights=[1.0] * self.env.num_coef
        )
        curr_pts = current_spline.evaluate(self.env.ts.reshape(-1, 1))
        
        # Store trajectory data
        centroid = np.mean(ctrl_pts, axis=0)
        self.trajectory_data.append(centroid.copy())
        self.reward_history.append(reward)
        dist = -reward
        self.distance_history.append(dist)
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize, facecolor=self.colors['background'])
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Main shape comparison plot
        ax_main = fig.add_subplot(gs[0, :2])
        self._plot_main_shapes(ax_main, curr_pts, ctrl_pts, step, dist, done)
        
        # Control point trajectory plot
        ax_traj = fig.add_subplot(gs[0, 2])
        self._plot_trajectory(ax_traj)
        
        # Metrics plots
        ax_reward = fig.add_subplot(gs[1, 0])
        self._plot_reward_history(ax_reward)
        
        ax_dist = fig.add_subplot(gs[1, 1])
        self._plot_distance_history(ax_dist)
        
        # Progress indicator
        ax_progress = fig.add_subplot(gs[1, 2])
        self._plot_progress_indicator(ax_progress, step, done)
        
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
        """Plot the main shape comparison"""
        # Create filled shapes for better visual contrast
        target_poly = patches.Polygon(self.target_pts, closed=True, 
                                    facecolor=self.colors['target'], 
                                    alpha=0.3, edgecolor=self.colors['target'], 
                                    linewidth=2, label='Target')
        current_poly = patches.Polygon(curr_pts, closed=True, 
                                     facecolor=self.colors['current'], 
                                     alpha=0.2, edgecolor=self.colors['current'], 
                                     linewidth=2, label='Current')
        
        ax.add_patch(target_poly)
        ax.add_patch(current_poly)
        
        # Plot control points with connections
        ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], 
                  c=self.colors['control'], s=60, zorder=5, 
                  edgecolors='white', linewidth=1.5, label='Control Points')
        
        # Connect control points with dashed lines
        for i in range(len(ctrl_pts)):
            next_i = (i + 1) % len(ctrl_pts)
            ax.plot([ctrl_pts[i, 0], ctrl_pts[next_i, 0]], 
                   [ctrl_pts[i, 1], ctrl_pts[next_i, 1]], 
                   '--', color=self.colors['control'], alpha=0.5, linewidth=1)
        
        # Add vectors showing deformation direction
        if len(self.trajectory_data) > 1:
            for i, ctrl_pt in enumerate(ctrl_pts):
                if i < len(self.prev_ctrl_pts):
                    dx = ctrl_pt[0] - self.prev_ctrl_pts[i, 0]
                    dy = ctrl_pt[1] - self.prev_ctrl_pts[i, 1]
                    if np.sqrt(dx**2 + dy**2) > 0.01:  # Only show significant movements
                        ax.arrow(self.prev_ctrl_pts[i, 0], self.prev_ctrl_pts[i, 1], 
                               dx, dy, head_width=0.05, head_length=0.05, 
                               fc=self.colors['trajectory'], ec=self.colors['trajectory'], 
                               alpha=0.7)
        
        self.prev_ctrl_pts = ctrl_pts.copy()
        
        # Styling
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(self.colors['background'])
        
        # Dynamic zoom based on shape bounds
        all_pts = np.vstack([self.target_pts, curr_pts])
        margin = 0.2
        x_range = [all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin]
        y_range = [all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin]
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # Status text with better formatting
        status = "CONVERGED!" if done and dist < 0.1 else "LEARNING..."
        status_color = self.colors['target'] if done and dist < 0.1 else self.colors['current']
        
        ax.text(0.02, 0.98, 
               f"Step: {step:2d}/{self.env.max_num_step}\n"
               f"Distance: {dist:.4f}\n"
               f"Status: {status}",
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                        edgecolor=status_color, alpha=0.9),
               fontsize=10, color=self.colors['text'], weight='bold')
        
        ax.legend(loc='upper right', framealpha=0.9)
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
    # Create environment
    env = ShapeBoundary(
        naive=False,
        step_size=5e-2,
        ctrl_state_dim=16,
        max_num_step=20,
        render_mode="rgb_array",
        degree=3,
        n_internal_knots=4,
    )
    
    # Initialize visualizer
    viz = EnhancedSplineVisualizer(env)
    viz.prev_ctrl_pts = np.zeros((env.num_coef, 2))  # Initialize previous control points
    
    # Reset environment and collect frames
    obs, _ = env.reset()
    frames = []
    
    # Create initial frame
    initial_frame = viz.create_frame(0, obs, 0.0)
    frames.append(initial_frame)
    
    # Run simulation
    for step in range(1, env.max_num_step + 1):
        # Use a more intelligent action that moves toward target
        # This creates a more interesting visualization than random actions
        action = generate_smart_action(env, obs)
        
        obs, reward, done, _, _ = env.step(action)
        frame = viz.create_frame(step, obs, reward, done)
        frames.append(frame)
        
        if done:
            # Add a few extra frames at the end to show final result
            for _ in range(3):
                frames.append(frame)
            break
    
    # Save as GIF
    imageio.mimsave("enhanced_spline_deformation.gif", frames, fps=3, loop=0)
    print("Created enhanced_spline_deformation.gif")
    
    return frames

def generate_smart_action(env, obs):
    """Generate a somewhat intelligent action that moves toward the target"""
    # Get current control points
    ctrl_flat = obs[:env.ctrl_dim]
    ctrl_pts = ctrl_flat.reshape(env.num_coef, 2)
    
    # Get target control points (approximate from target spline)
    target_angles = np.linspace(0, 2*np.pi, env.num_coef, endpoint=False)
    target_ctrl = np.stack([np.cos(target_angles), np.sin(target_angles)], axis=1) * 0.5
    
    # Create action that moves toward target
    diff = target_ctrl - ctrl_pts
    action_ctrl = diff.flatten() * 0.5  # Scale down the movement
    
    # Add some noise for more interesting dynamics
    noise = np.random.normal(0, 0.1, size=action_ctrl.shape)
    action_ctrl += noise
    
    # Handle knot dimensions (if any)
    if env.knot_dim > 0:
        knot_action = np.random.normal(0, 0.05, size=env.knot_dim)
        action = np.concatenate([action_ctrl, knot_action])
    else:
        action = action_ctrl
    
    # Clip to action space
    return np.clip(action, env.action_space.low, env.action_space.high)

if __name__ == "__main__":
    create_enhanced_visualization()