# helps visualize shape during training process
import os
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2


def write_video_frame(video_writer, frame, step_idx, reward=None, loss=None):
    """
    Writes a frame directly to a video writer with overlaid step and reward/loss information.
    """
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    overlay_text = f"Step: {step_idx}"
    if reward is not None:
        overlay_text += f" | Reward: {reward:.3f}"
    if loss is not None:
        overlay_text += f" | Loss: {loss:.3f}"

    draw.rectangle([5, 5, 5 + draw.textlength(overlay_text, font), 25], fill="white")
    draw.text((8, 8), overlay_text, fill="red", font=font)

    video_writer.append_data(np.array(frame_pil))


def plot_reward_curve(rewards, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(rewards, label='Reward')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def make_video_from_frames(frame_dir, output_path="evolution.mp4", fps=10):
    """
    (Legacy support) Compiles a folder of PNG frames into a single video.
    This method is no longer used if you stream frames directly using write_video_frame().
    """
    filenames = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".png")
    ])
    frames = [imageio.imread(f) for f in filenames]
    imageio.mimsave(output_path, frames, fps=fps)