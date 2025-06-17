import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from utils import setup_dpo_model
from benchmarks.sb3_utils import setup_benchmark_model

def visualize_util(method, env, env_name, num_step, extra_args, show_info=True, quality='high'):
    """Main visualization utility function with improved quality control."""
    if method.startswith('DPO'):
        model = setup_dpo_model(method, env, env_name)
        benchmark_model = False
    else:
        model = setup_benchmark_model(method, env, env_name)
        benchmark_model = True
    
    benchmark_str = '_benchmark' if benchmark_model else ''
    
    # Create output directory
    output_dir = Path('output/videos')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean filename
    clean_env_name = env_name.replace('.zip', '').replace('_', '-')
    img_path = output_dir / f'{clean_env_name}_{method}{benchmark_str}.mp4'
    print(f"Saving to: {img_path}")
    
    vals, images = visualize(
        env, model, 
        num_step=num_step, 
        benchmark_model=benchmark_model,
        extra_args=extra_args, 
        img_path=img_path,
        show_info=show_info,
        quality=quality
    )
    
    return vals, images

def visualize(env, model, num_step=100, benchmark_model=False,
              extra_args='random', img_path=None, show_info=True, quality='high'):
    """
    Enhanced visualization with proper image handling and quality controls.
    """
    vals = []
    rewards = []
    actions_history = []
    images = []
    cumulative_reward = 0
    
    # Reset environment
    try:
        obs = env.reset_at(mode=extra_args)
    except:
        obs, _ = env.reset()
    
    action = np.zeros(obs.shape) if hasattr(obs, 'shape') else np.zeros(1)
    
    print(f"Starting visualization: {num_step} steps")
    
    for step in range(num_step):
        # Get action from model
        try:
            if benchmark_model:
                action, _ = model.predict(obs)
            else:
                action = model.get_action(obs, action)
        except Exception as e:
            print(f"Action prediction error at step {step}: {e}")
            break
        
        # Step environment
        try:
            obs, reward, done, _, _ = env.step(action)
            cumulative_reward += reward
        except Exception as e:
            print(f"Environment step error at step {step}: {e}")
            break
        
        # Store metrics
        rewards.append(reward)
        actions_history.append(action.copy() if hasattr(action, 'copy') else action)
        
        # Render frame with error handling
        img = render_frame(env)
        if img is None:
            print(f"Failed to render frame at step {step}")
            continue
            
        # Process and enhance image
        img = process_image(img)
        
        # Add information overlay if requested
        if show_info:
            try:
                val = env.get_val(reward, action)
                img = add_enhanced_overlay(
                    img, step + 1, reward, cumulative_reward, val, action, done
                )
            except Exception as e:
                print(f"Overlay error at step {step}: {e}")
                # Add minimal overlay as fallback
                img = add_minimal_overlay(img, step + 1, reward, cumulative_reward, done)
        
        images.append(img)
        
        try:
            vals.append(env.get_val(reward, action))
        except:
            vals.append(reward)  # Fallback to reward if get_val fails
        
        if done:
            print(f"Episode finished at step {step + 1}")
            break
    
    # Close environment
    try:
        env.close()
    except:
        pass
    
    # Save high-quality video
    if img_path is not None and images:
        success = save_video(images, img_path, quality=quality)
        if success:
            print(f"✓ Video saved: {img_path}")
            print(f"  Frames: {len(images)}, Quality: {quality}")
            print(f"  Final reward: {cumulative_reward:.3f}")
        else:
            print(f"✗ Failed to save video: {img_path}")
    
    return vals, images

def render_frame(env):
    """Safely render a frame from the environment."""
    try:
        # Try different rendering methods
        if hasattr(env, 'render'):
            # Try with rgb_array mode first
            try:
                img = env.render(mode='rgb_array')
                if img is not None:
                    return img
            except:
                pass
            
            # Try without mode parameter
            try:
                img = env.render()
                if img is not None:
                    return img
            except:
                pass
        
        # If render fails, try getting screen
        if hasattr(env, 'get_screen'):
            try:
                return env.get_screen()
            except:
                pass
                
        return None
        
    except Exception as e:
        print(f"Render error: {e}")
        return None

def process_image(img):
    """Process and enhance the raw image from environment."""
    if img is None:
        return create_fallback_image()
    
    # Convert to numpy array
    img = np.array(img)
    
    # Handle different image formats
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3 and img.dtype != np.uint8:  # RGB but wrong dtype
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # Ensure proper data type
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Enhance image quality
    img = enhance_image_quality(img)
    
    return img

def enhance_image_quality(img):
    """Enhance image quality with basic processing."""
    # Resize if too small (for better visibility)
    h, w = img.shape[:2]
    if h < 200 or w < 200:
        scale_factor = max(208 / h, 208 / w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Apply slight sharpening for better clarity
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel * 0.1 + np.eye(3, dtype=np.float32) * 0.9)
    
    # Ensure values are in valid range
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

def create_fallback_image(size=(400, 400)):
    """Create a fallback image when rendering fails."""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 50  # Dark gray
    cv2.putText(img, "Render Failed", (size[0]//2 - 80, size[1]//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def add_enhanced_overlay(img, step, reward, cumulative_reward, val, action, done):
    """Add enhanced information overlay with better styling."""
    h, w = img.shape[:2]
    img_overlay = img.copy()
    
    # Create semi-transparent panels
    overlay = img_overlay.copy()
    
    # Top panel
    panel_height = min(100, h // 4)
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    
    # Bottom panel for action info
    bottom_panel_height = min(60, h // 6)
    cv2.rectangle(overlay, (0, h - bottom_panel_height), (w, h), (0, 0, 0), -1)
    
    # Blend panels
    alpha = 0.75
    img_overlay = cv2.addWeighted(img_overlay, 1 - alpha, overlay, alpha, 0)
    
    # Text styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(0.7, w / 800)  # Scale text based on image width
    thickness = max(1, int(font_scale * 2))
    
    # Colors (BGR)
    colors = {
        'white': (255, 255, 255),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'orange': (0, 165, 255),
        'purple': (255, 0, 255)
    }
    
    # Dynamic text positioning
    y_offset = max(25, panel_height // 4)
    
    # Main metrics
    texts_and_colors = [
        (f"Step: {step}", colors['white']),
        (f"Reward: {reward:.1f}", colors['green'] if reward > 0 else colors['red'] if reward < 0 else colors['white']),
        (f"Total: {cumulative_reward:.1f}", colors['green'] if cumulative_reward > 0 else colors['red']),
        (f"Value: {val:.1f}", colors['cyan'])
    ]
    
    # Draw texts with dynamic spacing (avoid overlap)
    x_cursor = 10  # starting x
    for text, color in texts_and_colors:
        cv2.putText(img_overlay, text, (x_cursor, y_offset),
                    font, font_scale, color, thickness)
        text_width, _ = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x_cursor += text_width + 5  # spacing between metrics
    
    # Action information
    action_text = format_action_text(action)
    cv2.putText(img_overlay, action_text, (10, h - bottom_panel_height + 30), 
               font, font_scale * 0.8, colors['yellow'], thickness)
    
    return img_overlay

def format_action_text(action):
    """Format action for display."""
    try:
        if hasattr(action, '__len__') and len(action) > 1:
            if len(action) <= 4:
                return f"Action: [{', '.join([f'{a:.2f}' for a in action])}]"
            else:
                return f"Action: [{', '.join([f'{a:.2f}' for a in action[:3]])}...] (||a||={np.linalg.norm(action):.2f})"
        else:
            return f"Action: {float(action):.3f}"
    except:
        return f"Action: {str(action)[:20]}"

def add_minimal_overlay(img, step, reward, cumulative_reward, done):
    """Minimal overlay for when detailed overlay fails."""
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # Simple top bar
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    alpha = 0.7
    img_overlay = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    # Simple text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 0) if cumulative_reward >= 0 else (0, 0, 255)
    
    text = f"Step:{step} R:{reward:.2f} Total:{cumulative_reward:.2f}"
    if done:
        text += " [DONE]"
    
    cv2.putText(img_overlay, text, (5, 20), font, font_scale, color, thickness)
    
    return img_overlay

def save_video(images, output_path, fps=30, quality='high'):
    """Save video with multiple fallback options for maximum compatibility."""
    if not images:
        print("No images to save!")
        return False
    
    # Ensure consistent image dimensions
    images = standardize_images(images)
    
    # Quality settings
    quality_configs = {
        'low': {'bitrate': '500k', 'crf': 28, 'preset': 'fast'},
        'medium': {'bitrate': '1500k', 'crf': 23, 'preset': 'medium'},
        'high': {'bitrate': '3000k', 'crf': 18, 'preset': 'slow'},
        'ultra': {'bitrate': '8000k', 'crf': 15, 'preset': 'veryslow'}
    }
    
    config = quality_configs.get(quality, quality_configs['high'])
    output_path = str(output_path)
    
    # Try multiple encoding methods
    methods = [
        ('ffmpeg_writer', save_with_ffmpeg_writer),
        ('imageio_mp4', save_with_imageio_mp4),
        ('opencv', save_with_opencv),
        ('imageio_gif', save_with_imageio_gif)
    ]
    
    for method_name, save_func in methods:
        try:
            print(f"Trying {method_name}...")
            success = save_func(images, output_path, fps, config)
            if success:
                print(f"✓ Saved with {method_name}")
                return True
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
            continue
    
    print("All save methods failed!")
    return False

def standardize_images(images):
    """Ensure all images have the same dimensions and format."""
    if not images:
        return images
    
    # Find the most common dimensions
    shapes = [img.shape[:2] for img in images]
    target_shape = max(set(shapes), key=shapes.count)
    
    standardized = []
    for img in images:
        img = np.array(img, dtype=np.uint8)
        
        # Resize if needed
        if img.shape[:2] != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), 
                           interpolation=cv2.INTER_AREA)
        
        # Ensure RGB format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        standardized.append(img)
    
    return standardized

def save_with_ffmpeg_writer(images, output_path, fps, config):
    """Save using imageio's ffmpeg writer."""
    output_path = output_path.replace('.wmv', '.mp4')
    
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        bitrate=config['bitrate'],
        pixelformat='yuv420p',
        ffmpeg_params=['-preset', config['preset'], '-crf', str(config['crf'])]
    )
    
    for img in images:
        writer.append_data(img)
    writer.close()
    return True

def save_with_imageio_mp4(images, output_path, fps, config):
    """Save using basic imageio MP4."""
    output_path = output_path.replace('.wmv', '.mp4')
    imageio.mimsave(output_path, images, fps=fps, format='mp4')
    return True

def save_with_opencv(images, output_path, fps, config):
    """Save using OpenCV VideoWriter."""
    output_path = output_path.replace('.wmv', '.mp4')
    
    if not images:
        return False
    
    h, w, c = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img in images:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    out.release()
    return True

def save_with_imageio_gif(images, output_path, fps, config):
    """Fallback: save as GIF."""
    output_path = output_path.replace('.mp4', '.gif').replace('.wmv', '.gif')
    imageio.mimsave(output_path, images, fps=min(fps, 10), format='gif')
    print(f"Saved as GIF: {output_path}")
    return True

def test_model_through_vals(seeds, env, model, num_traj, num_step_per_traj,
                           benchmark_model=False):
    """Run trajectories and collect values with better error handling."""
    all_vals = [[] for _ in range(len(seeds))]
    
    for i, seed in enumerate(seeds):
        try:
            env.rng = np.random.default_rng(seed=seed)
        except:
            np.random.seed(seed)
        
        for traj in range(num_traj):
            try:
                obs, _ = env.reset()
                action = np.zeros(obs.shape) if hasattr(obs, 'shape') else np.zeros(1)
                vals_cur_traj = []
                
                for step in range(num_step_per_traj):
                    if benchmark_model:
                        action, _ = model.predict(obs)
                    else:
                        action = model.get_action(obs, action)
                    
                    obs, reward, done, _, _ = env.step(action)
                    
                    try:
                        val = env.get_val(reward, action)
                    except:
                        val = reward
                    
                    vals_cur_traj.append(val)
                    
                    if done:
                        break
                
                all_vals[i].append(np.array(vals_cur_traj))
                
            except Exception as e:
                print(f"Error in trajectory {traj} for seed {seed}: {e}")
                continue
    
    # Convert to array with proper padding
    max_len = max(len(traj) for seed_vals in all_vals for traj in seed_vals) if any(all_vals) else num_step_per_traj
    
    result = []
    for seed_vals in all_vals:
        for traj_vals in seed_vals:
            padded = np.zeros(max_len)
            padded[:len(traj_vals)] = traj_vals
            result.append(padded)
    
    return np.array(result)

# Convenience functions for different visualization types
def visualize_clean(method, env, env_name, num_step, extra_args):
    """Clean visualization without overlays."""
    return visualize_util(method, env, env_name, num_step, extra_args, 
                         show_info=False, quality='high')

def visualize_detailed(method, env, env_name, num_step, extra_args):
    """Detailed visualization with full overlays."""
    return visualize_util(method, env, env_name, num_step, extra_args, 
                         show_info=True, quality='ultra')

def quick_visualize(method, env, env_name, num_step, extra_args):
    """Quick visualization with lower quality for testing."""
    return visualize_util(method, env, env_name, num_step, extra_args, 
                         show_info=True, quality='low')