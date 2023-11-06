import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def point_cloud_to_birds_eye_view(points, res=0.1, side_range=(-20, 20), fwd_range=(-20, 20)):
    """Converts a 3D point cloud into a bird's eye view map.
    
    Args:
        points: numpy array of shape (N, 3) containing the x, y, and z coordinates of the points
        res: float, resolution of the grid in meters per pixel (default: 0.1)
        side_range: tuple of floats, limits of the side (y) coordinates of the grid in meters (default: (-20, 20))
        fwd_range: tuple of floats, limits of the forward (x) coordinates of the grid in meters (default: (-20, 20))
    
    Returns:
        numpy array of shape (H, W) containing the bird's eye view map, where H and W are determined by the
        res and side_range/fwd_range arguments
    """
    # Define the grid
    W = int((side_range[1] - side_range[0]) / res)
    H = int((fwd_range[1] - fwd_range[0]) / res)
    grid = np.zeros((H, W), dtype=np.float32)

    # Project the points onto the 2D plane perpendicular to the ground
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    x_img = (-y_points / res).astype(np.int32) + int(W / 2)
    y_img = (-x_points / res).astype(np.int32) + int(H / 2)
    x_img = np.clip(x_img, 0, W - 1)
    y_img = np.clip(y_img, 0, H - 1)
    z_min = np.percentile(z_points, 0.1)
    z_max = np.percentile(z_points, 99.9)
    pixel_values = np.clip((z_points - z_min) / (z_max - z_min), 0, 1)

    # Map the points onto the grid
    grid[y_img, x_img] = pixel_values

    return grid

def nuscenes_box_to_birds_eye_view(box, pose_record, calib, res=0.1, side_range=(-20, 20), fwd_range=(-20, 20)):
    """Converts a NuScenes 3D box into a bird's eye view box that can be plotted on top of the lidar map.
    
    Args:
        box: NuScenes Box instance containing the 3D box coordinates and dimensions
        pose_record: NuScenes SamplePose instance containing the pose of the ego vehicle
        calib: NuScenes Calibration instance containing the sensor calibration data
        res: float, resolution of the grid in meters per pixel (default: 0.1)
        side_range: tuple of floats, limits of the side (y) coordinates of the grid in meters (default: (-20, 20))
        fwd_range: tuple of floats, limits of the forward (x) coordinates of the grid in meters (default: (-20, 20))
    
    Returns:
        numpy array of shape (4,) containing the 2D box coordinates in the form [x_min, y_min, x_max, y_max]
    """
    # Get the rotation matrix and translation vector of the box
    rotation_matrix = box.orientation.rotation_matrix
    translation_vector = box.center

    # Convert the box coordinates to the ego vehicle frame
    translation_vector -= pose_record.translation
    translation_vector = translation_vector @ np.linalg.inv(pose_record.rotation)
    rotation_matrix = rotation_matrix @ np.linalg.inv(pose_record.rotation)

    # Convert the box coordinates to the sensor frame
    translation_vector -= calib.sensor_transform.translation
    rotation_matrix = calib.sensor_transform.rotation_matrix @ rotation_matrix
    translation_vector = calib.sensor_transform.rotation_matrix @ translation_vector

    # Project the box onto the 2D plane perpendicular to the ground
    corners = box.corners()
    x_points = corners[:, 0]
    y_points = corners[:, 1]
    z_points = corners[:, 2]
    x_img = (-y_points / res).astype(np.int32) + int((side_range[1] - side_range[0]) / res / 2)
    y_img = (-x_points / res).astype(np.int32) + int((fwd_range[1] - fwd_range[0]) / res / 2)
    x_img = np.clip(x_img, 0, int((side_range[1] - side_range[0]) / res) - 1)
    y_img = np.clip(y_img, 0, int((fwd_range[1] - fwd_range[0]) / res) - 1)
    z_min = np.percentile(z_points, 0.1)
    z_max = np.percentile(z_points, 99.9)

    # Extract the 2D box coordinates
    x_min = np.min(x_img)
    y_min = np.min(y_img)
    x_max = np.max(x_img)
    y_max = np.max(y_img)

    return np.array([x_min, y_min, x_max, y_max])


# Load a point cloud from a PLY file
point_cloud = o3d.io.read_point_cloud('point_cloud.ply')
points = np.asarray(point_cloud.points)

# Convert the point cloud to a bird's eye view map
grid = point_cloud_to_birds_eye_view(points)

# Display the bird's eye view map
plt.imshow(grid, cmap='gray')
plt.show()
