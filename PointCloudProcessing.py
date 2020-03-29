import open3d
import numpy as np


class PointCloudProcessing:
    """
    Class for point cloud processing needs
    """

    def __init__(self):
        pass

    def process(self, point_cloud_left: open3d.geometry.PointCloud, point_cloud_right: open3d.geometry.PointCloud):
        downsampled_left = self.preprocess_point_cloud(point_cloud_left)
        downsampled_right = self.preprocess_point_cloud(point_cloud_right)
        transformation = self.multiscale_icp(downsampled_left, downsampled_right,
                                             [5, 4, 3, 2, 1, 0.5], [50, 45, 40, 35, 30])
        downsampled_left.transform(transformation)
        open3d.visualization.draw_geometries([downsampled_left, downsampled_right])

    def preprocess_point_cloud(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        # points = np.asarray(point_cloud.points)
        # mask = points[:, 1] > -700
        # point_cloud.points = open3d.utility.Vector3dVector(points[mask])
        # point_cloud.colors = open3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[mask])
        downsampled = self.downsample(point_cloud)
        downsampled = self.background_removal(downsampled)
        downsampled, inliers = downsampled.remove_radius_outlier(nb_points=60, radius=12)
        return downsampled

    def downsample(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=1.7)
        return voxel_down_pcd.uniform_down_sample(every_k_points=2)

    def background_removal(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        plane, inliers = point_cloud.segment_plane(distance_threshold=0.005, ransac_n=4, num_iterations=250)
        return point_cloud.select_down_sample(inliers, invert=True)

    def multiscale_icp(self, source, target, voxel_size, max_iter, init_transformation=np.identity(4)):
        # current_transformation = init_transformation
        current_transformation = [[9.70146379e-01, - 3.94297931e-02, - 2.39293325e-01, 1.37356105e+02],
                                  [4.12079500e-02, 9.99147636e-01, 2.43031811e-03, - 3.41151242e+01],
                                  [2.38993533e-01, - 1.22185517e-02, 9.70944282e-01, - 7.84797055e+01],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
            iter = max_iter[scale]
            distance_threshold = voxel_size[i] * 1.4
            source_down = source.voxel_down_sample(voxel_size[scale])
            target_down = target.voxel_down_sample(voxel_size[scale])

            source_down.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                        max_nn=30))
            target_down.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                        max_nn=30))

            result_icp = open3d.registration.registration_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                open3d.registration.TransformationEstimationPointToPlane(),
                open3d.registration.ICPConvergenceCriteria(max_iteration=iter))

            current_transformation = result_icp.transformation

        return result_icp.transformation


if __name__ == "__main__":
    point_cloud_processing = PointCloudProcessing()
    pcl_left = open3d.io.read_point_cloud("pcl_left.pcd")
    pcl_right = open3d.io.read_point_cloud("pcl_right.pcd")
    point_cloud_processing.process(pcl_left, pcl_right)
    # point_cloud_downsampled = point_cloud_processing.preprocess_point_cloud(pcl_right)
    # open3d.visualization.draw_geometries([point_cloud_downsampled])
    # transformation, information = point_cloud_processing.multiscale_icp(pcl_left, pcl_right, [voxel_size, voxel_size / 2, voxel_size / 4], [50, 30, 14])
