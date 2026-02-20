import pybullet as p
import pybullet_data
import numpy as np
import os
from icrl.constraint_net import ConstraintNet
import time
from icrl import utils


def main():
    p.connect(p.GUI)
    p.resetSimulation()
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering during setup

    urdfRootPath = pybullet_data.getDataPath()
    p.setGravity(0, 0, -9.81)
    planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
    # pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
    tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
    #
    # rest_poses = [0, -0.215, 0., -3., 0, 2.356, 2.356, 0.08, 0.08]
    # for i in range(7):
    #     p.resetJointState(pandaUid, i, rest_poses[i])
    # p.resetJointState(pandaUid, 9, 0.08)
    # p.resetJointState(pandaUid, 10, 0.08)
    #
    # # Set starting point
    # starting_point = (random.uniform(0.35, 0.35), random.uniform(-0.0, 0.0), random.uniform(0.2, 0.2))
    # orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
    # rest_poses = p.calculateInverseKinematics(pandaUid, 11, starting_point, orientation, maxNumIterations=300, residualThreshold=.003)[0:7]
    # for i in range(7):
    #     p.resetJointState(pandaUid, i, rest_poses[i])

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Enable rendering

    collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0001)
    visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[0, 0, 1, 1])  # Red small sphere

    def setup_obstacles_concave():
        for obstacle_position in np.array([[0.55, 0.0, 0], [0.52, -0.0512, 0], [0.515, 0.052, 0], [0.477, 0.103, 0]]):
            p.loadURDF("gym_panda/envs/cup/cup_small.urdf", basePosition=obstacle_position)

    def setup_obstacles_2regions():
        for obstacle_position in np.array([[0.55, -0.04, 0], [0.51, -0.0912, 0], [0.515, 0.072, 0], [0.477, 0.123, 0]]):
            p.loadURDF("gym_panda/envs/cup/cup_small.urdf", basePosition=obstacle_position)

    def plot_demonstrations(path):
        # Plot demonstrations
        (obs_e, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(path, 20)
        trajectory_start_idx = 0
        for length in expert_lengths[:17]:
            trajectory_end_idx = trajectory_start_idx + length
            target = obs_e[trajectory_end_idx - 1, :3]

            i = trajectory_start_idx
            if np.linalg.norm(target - obs_e[trajectory_start_idx + 1][:3]) > 0.1:
                while i < trajectory_end_idx - 1:
                    start_point = obs_e[i][:3]
                    end_point = obs_e[i + 1][:3]
                    if np.linalg.norm(start_point - target) <= 0.01:
                        break
                    if np.linalg.norm(start_point - end_point) <= 0.03:
                        p.addUserDebugLine(start_point, end_point, lineColorRGB=[0, 0.8, 0], lineWidth=3.0)
                    i += 1
            trajectory_start_idx = trajectory_end_idx

    def plot_rollouts(path):
        # Plot policy rollouts
        colors = [
            [1, 0.65, 0],  # Orange
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 0, 0],  # Red
            [0, 1, 1],  # Cyan
            [1, 0, 1],  # Magenta
            [0.5, 0, 0.5],  # Purple
            [0.7, 0.7, 0.2]  # Light Yellow
        ]

        (obs, _, _), lengths, _ = utils.load_expert_data(path, 20, load_type='transfer')
        color_index = 0
        current_obs_index = 0
        i = 0

        while i < obs.shape[0] - 1:
            start_point = obs[i][:3]
            end_point = obs[i + 1][:3]
            target = obs[current_obs_index + lengths[color_index] - 1][:3]  # Update target for each trajectory

            if i >= current_obs_index + lengths[color_index]:
                color_index = (color_index + 1) % len(colors)
                current_obs_index += lengths[color_index]

            if np.linalg.norm(start_point - end_point) > 0.1:
                i += 1
                continue

            if np.linalg.norm(start_point - target) <= 0.03:
                i = current_obs_index + lengths[color_index]  # Jump to the next trajectory start
                continue

            p.addUserDebugLine(start_point, end_point, lineColorRGB=colors[color_index], lineWidth=3.0)
            i += 1

    def plot_constraint(path):
        # Plot learned constraints
        constraint_net = ConstraintNet.load(path)
        x_lim = (0.42, 0.6)
        y_lim = (-0.12, 0.18)
        z_lim = (0.02, 0.2)
        num_points_for_plot = 18
        cost_function = constraint_net.cost_function

        r1 = np.linspace(x_lim[0], x_lim[1], num=num_points_for_plot)
        r2 = np.linspace(y_lim[0], y_lim[1], num=num_points_for_plot)
        r3 = np.linspace(z_lim[0], z_lim[1], num=10)
        X, Y, Z = np.meshgrid(r1, r2, r3)

        obs_dim = 9
        acs_dim = 3
        obs = np.concatenate(
            [X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1]), np.zeros((X.size, obs_dim - 3))], axis=-1)
        action = np.zeros((np.size(X), acs_dim))
        outs = (1 - cost_function(obs, action)).reshape(X.shape)

        mask = outs < 0.5
        X_masked, Y_masked, Z_masked, outs_masked = X[mask], Y[mask], Z[mask], outs[mask]
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.007, rgbaColor=[1, 0, 0, 0.7])  # Red small sphere
        for i in range(X_masked.shape[0]):
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id,
                              basePosition=[X_masked[i], Y_masked[i], Z_masked[i]])

    def set_camera():
        # camera_distance = 1.
        # camera_yaw = 40
        # camera_pitch = -71
        # camera_target_position = [0.44, 0.131, -0.493]
        camera_distance = 1.
        camera_yaw = 38
        camera_pitch = -73
        camera_target_position = [0.44, 0.131, -0.493]
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    # Function to print current camera parameters
    def print_current_camera_parameters():
        cam_info = p.getDebugVisualizerCamera()
        print("Camera Distance:", cam_info[10])
        print("Camera Yaw:", cam_info[8])
        print("Camera Pitch:", cam_info[9])
        print("Camera Target Position:", cam_info[11])

    # Plotting for concave
    # utils.load_expert_data_and_plot(f'icrl/wandb/run-20240716_093851-0lehfict', 20, load_type='transfer')

    set_camera()
    # Plotting for concave 
    setup_obstacles_concave()
    plot_demonstrations('icrl/expert_data/ReachConcaveObsDS')
    # plot_constraint(f'icrl/wandb/run-20240716_093851-0lehfict/files/models/icrl_19_itrs/cn.pt')
    # plot_rollouts(f'icrl/wandb/run-20240716_093851-0lehfict')


    # Plotting for 2regions
    # setup_obstacles_2regions()
    # utils.load_expert_data_and_plot('icrl/expert_data/Reach2RegionObsDS', 222)
    # plot_demonstrations('icrl/expert_data/Reach2RegionObsDS')
    # plot_rollouts(f'icrl/wandb/run-20240718_041918-4svsme3b')
    # plot_constraint(f'icrl/wandb/run-20240718_041918-4svsme3b/files/models/icrl_6_itrs/cn.pt')

    while True:
        print_current_camera_parameters()
        p.stepSimulation()
        time.sleep(1)

if __name__ == "__main__":
    main()