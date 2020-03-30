import logging

import open3d
import numpy as np


class PointCloudProcessing:
    """
    Class for point cloud processing needs
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def process(self, point_cloud_left: open3d.geometry.PointCloud, point_cloud_right: open3d.geometry.PointCloud):
        self.logger.info("Preprocessing...")
        downsampled_left = self.preprocess_point_cloud(point_cloud_left)
        downsampled_right = self.preprocess_point_cloud(point_cloud_right)
        self.logger.info("Merging...")
        transformation = self.multiscale_icp(downsampled_left, downsampled_right,
                                             [6, 5, 4, 3, 2, 1, 0.5], [60, 50, 45, 40, 35, 30])
        downsampled_left.transform(transformation)
        final_point_cloud = self.post_process(downsampled_left, downsampled_right)
        self.logger.info("Creating mesh...")
        return self.convert_to_mesh(final_point_cloud)

    def preprocess_point_cloud(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        downsampled = self.downsample(point_cloud)
        downsampled = self.background_removal(downsampled)
        downsampled, inliers = downsampled.remove_radius_outlier(nb_points=60, radius=12)
        return downsampled

    def post_process(self, point_cloud_left: open3d.geometry.PointCloud, point_cloud_right: open3d.geometry.PointCloud):
        final_point_cloud = point_cloud_left + point_cloud_right
        final_point_cloud = self.downsample(final_point_cloud)
        return final_point_cloud

    def convert_to_mesh(self, point_cloud: open3d.geometry.PointCloud):
        point_cloud.estimate_normals()
        distances = point_cloud.compute_nearest_neighbor_distance()
        average_distance = np.mean(distances)
        radius = 1.5 * average_distance
        mesh = open3d.geometry.TriangleMesh()
        mesh = mesh.create_from_point_cloud_ball_pivoting(pcd=point_cloud, radii=open3d.utility.DoubleVector([radius, radius * 2]))
        mesh = mesh.filter_smooth_simple()
        mesh = mesh.compute_vertex_normals()
        return mesh

    def downsample(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=1.7)
        return voxel_down_pcd.uniform_down_sample(every_k_points=2)

    def background_removal(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        plane, inliers = point_cloud.segment_plane(distance_threshold=0.005, ransac_n=4, num_iterations=250)
        return point_cloud.select_down_sample(inliers, invert=True)

    def multiscale_icp(self, source, target, voxel_size, max_iter, init_transformation=np.identity(4)):
        current_transformation = init_transformation
        source = self.crop_y(source)
        target = self.crop_y(target)

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

    def crop_y(self, point_cloud):
        points = np.asarray(point_cloud.points)
        mask = points[:, 1] > -700
        point_cloud_cropped = open3d.geometry.PointCloud()
        point_cloud_cropped.points = open3d.utility.Vector3dVector(points[mask])
        point_cloud_cropped.colors = open3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[mask])
        return point_cloud_cropped


if __name__ == "__main__":
    point_cloud_processing = PointCloudProcessing()
    pcl_left = open3d.io.read_point_cloud("pcl_left.pcd")
    pcl_right = open3d.io.read_point_cloud("pcl_right.pcd")
    pcl_final = point_cloud_processing.process(pcl_left, pcl_right)
    open3d.visualization.draw_geometries([pcl_final])
