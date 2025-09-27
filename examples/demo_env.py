"""
Example usage: 

python examples/demo_env.py --env_name SquareWide

For <Task>Real envs, we support composite robots and specify them using i) the --composite-robot flag
and ii) the naming scheme of <robot>_<gripper>:

python examples/demo_env.py --env_name SquareReal --robots Panda_PandaUmiGripper --composite-robot
"""
import argparse
from collections import defaultdict
import numpy as np
import imageio
from pathlib import Path
import time

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_real_depth_map

import cpgen_envs


def parse_args():
    parser = argparse.ArgumentParser(description="Robosuite environment runner with argparse support.")
    parser.add_argument("--env_name", type=str, default="SquareWide", help="Name of the environment")
    parser.add_argument("--robots", nargs="+", default="Panda", help="List of robots to use")
    # for composite robot, use <robot>_<gripper> e.g. Panda_PandaUmiGripper
    parser.add_argument("--composite-robot", action="store_true", help="Create composite robot based off robot and gripper")
    parser.add_argument("--cam_h", type=int, default=720, help="Height of the camera")
    parser.add_argument("--cam_w", type=int, default=1280, help="Width of the camera")
    parser.add_argument("--camera_names", nargs="+", default=["agentview"], help="List of camera names to use")
    parser.add_argument("--save-xml", action="store_true", help="Save the environment xml to disk")
    parser.add_argument("--save-rgb-depth-seg", action="store_true", help="Save RGB, depth and segmentation images")
    parser.add_argument("--renderer", type=str, default="mujoco",
        help="Renderer to use. mujoco: offscreen rendering. mjviewer: on-screen rendering.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Print welcome info
    print(f"Welcome to robosuite v{suite.__version__}!")
    print(suite.__logo__)
    
    # Create options dictionary
    options = {
        "env_name": args.env_name,
        "env_configuration": "single-arm-parallel",
        "robots": args.robots,
    }
    
    if args.composite_robot:
        options["robots"] = args.robots[0]

    # Load the desired controller
    options["controller_configs"] = load_composite_controller_config(
        controller=None,  # None for default
        robot=options["robots"][0],
    )

    if args.renderer == "mujoco":
        has_offscreen_renderer = True
        use_camera_obs = True
        has_renderer = False
    elif args.renderer == "mjviewer":
        has_offscreen_renderer = False
        use_camera_obs = False
        has_renderer = True
    # Initialize the environment
    env = suite.make(
        **options,
        renderer=args.renderer,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        ignore_done=True,
        use_camera_obs=use_camera_obs,
        control_freq=20,
        camera_heights=args.cam_h,
        camera_widths=args.cam_w,
        camera_names=args.camera_names,
        camera_segmentations="instance",
        camera_depths=True
    )

    Path("examples/demo_images/").mkdir(parents=True, exist_ok=True)

    if args.save_xml:
        env.reset()
        # get env xml and save to disk
        model_xml = env.sim.model.get_xml()
        with open(f"examples/demo_images/{args.env_name}.xml", "w") as f:
            f.write(model_xml)
        print(f"Saved xml to examples/demo_images/{args.env_name}.xml")

    if args.renderer == "mjviewer":
        env.reset()
        for _ in range(10000):
            env.step(np.ones(7))
            env.render()
            time.sleep(0.01)

    # Initialize video writer
    frames = defaultdict(list)
    for i in range(5):
        obs = env.reset()

        for camera_name in args.camera_names:
            # Save PNG
            imageio.imwrite("examples/demo_images/" + f"{args.env_name}-{camera_name}-{i}.png", obs[f"{camera_name}_image"][::-1])
            print(f"Saved image to examples/demo_images/{args.env_name}-{camera_name}-{i}.png")
            # check segmentation mask
            if obs.get(f"{camera_name}_segmentation_instance", None) is not None:
                # save binary segmentation mask
                seg_mask = obs[f"{camera_name}_segmentation_instance"][::-1].squeeze()
                seg_mask[seg_mask != 0] = 1
                seg_mask = seg_mask.astype(np.uint8) * 255
                imageio.imwrite("examples/demo_images/" + f"{args.env_name}-{camera_name}-{i}_seg.png", seg_mask)
                print(f"Saved segmentation mask to examples/demo_images/{args.env_name}-{camera_name}-{i}_seg.png")
            # check depth
            if obs.get(f"{camera_name}_depth", None) is not None:
                # save depth image
                depth = obs[f"{camera_name}_depth"]
                real_depth = get_real_depth_map(env.sim, depth)
                real_depth = real_depth.clip(0, 1.75)[::-1].squeeze()
                real_depth = (real_depth / 1.75 * 255).astype(np.uint8)
                imageio.imwrite("examples/demo_images/" + f"{args.env_name}-{camera_name}-{i}_depth.png", real_depth)
                print(f"Saved depth image to examples/demo_images/{args.env_name}-{camera_name}-{i}_depth.png")

            frames[f"{camera_name}"].append(obs[f"{camera_name}_image"][::-1])

            # Combine RGB, depth and segmentation as horizontal image if all exist
            if args.save_rgb_depth_seg and all(key in obs for key in [f"{camera_name}_image", f"{camera_name}_depth", f"{camera_name}_segmentation_instance"]):
                # Prepare images for combining
                rgb_img = obs[f"{camera_name}_image"][::-1]
                
                depth = obs[f"{camera_name}_depth"]
                real_depth = get_real_depth_map(env.sim, depth)
                real_depth = real_depth.clip(0, 1.75)[::-1].squeeze()
                depth_img = (real_depth / 1.75 * 255).astype(np.uint8)
                # Convert to 3 channels for concatenation
                depth_img = np.stack([depth_img] * 3, axis=2)
                
                seg_mask = obs[f"{camera_name}_segmentation_instance"][::-1].squeeze()
                seg_mask[seg_mask != 0] = 1
                seg_img = (seg_mask.astype(np.uint8) * 255)
                # Convert to 3 channels for concatenation
                seg_img = np.stack([seg_img] * 3, axis=2)
                
                # Horizontal concatenation
                combined_img = np.concatenate([rgb_img, depth_img, seg_img], axis=1)
                
                # Save combined image
                combined_path = f"examples/demo_images/{args.env_name}-{camera_name}-{i}_combined.png"
                imageio.imwrite(combined_path, combined_img)
                print(f"Saved combined image to {combined_path}")

        env.step(np.ones(7))

    for camera_name in args.camera_names:
        frames[camera_name] = np.array(frames[camera_name])
        print(f"Camera {camera_name} has {len(frames[camera_name])} frames")
        video_path = f"examples/demo_images/{args.env_name}-{camera_name}.mp4"
        imageio.mimsave(video_path, frames[camera_name], fps=2)
        print(f"Saved video to {video_path}")

    print(f"Running {args.env_name} environment with {args.robots} robot")
    # Get action limits
    low, high = env.action_spec
    
    # Do visualization
    for _ in range(100):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
