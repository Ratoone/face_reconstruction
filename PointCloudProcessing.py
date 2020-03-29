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
        # return self.icp_registration(downsampled_left, downsampled_right)
        voxel_size = 3
        transformation, _ = self.multiscale_icp(downsampled_left, downsampled_right, [voxel_size, voxel_size / 2, voxel_size / 4], [50, 30, 14])
        downsampled_left.transform(transformation)
        open3d.visualization.draw_geometries([downsampled_left, downsampled_right])

    def icp_registration(self, point_cloud_left: open3d.geometry.PointCloud,
                         point_cloud_right: open3d.geometry.PointCloud):
        transformation_matrix = np.eye(4)
        evaluation = open3d.registration.evaluate_registration(point_cloud_left, point_cloud_right, 0.02,
                                                               transformation_matrix)
        print(evaluation)
        registration = open3d.registration.registration_icp(point_cloud_left,
                                                            point_cloud_right,
                                                            1e-3,
                                                            transformation_matrix,
                                                            open3d.registration.TransformationEstimationPointToPoint(),
                                                            open3d.registration.ICPConvergenceCriteria(
                                                                max_iteration=300)
                                                            )
        print(registration.transformation)
        print(registration)
        return None

    def preprocess_point_cloud(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        downsampled = self.downsample(point_cloud)
        downsampled = self.background_removal(downsampled)
        downsampled, inliers = downsampled.remove_radius_outlier(nb_points=12, radius=5)
        downsampled, inliers = downsampled.remove_radius_outlier(nb_points=8, radius=5)
        return downsampled

    def downsample(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=0.8)
        return voxel_down_pcd.uniform_down_sample(every_k_points=4)

    def background_removal(self, point_cloud: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        plane, inliers = point_cloud.segment_plane(distance_threshold=0.005, ransac_n=4, num_iterations=250)
        return point_cloud.select_down_sample(inliers, invert=True)

    def multiscale_icp(self, source, target, voxel_size, max_iter, init_transformation=np.identity(4)):
        current_transformation = init_transformation
        for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
            iter = max_iter[scale]
            distance_threshold = voxel_size[i] * 1.4
            print("voxel_size %f" % voxel_size[scale])
            source_down = source.voxel_down_sample(voxel_size[scale])
            target_down = target.voxel_down_sample(voxel_size[scale])

            # result_icp = open3d.registration.registration_icp(
            #     source_down, target_down, distance_threshold,
            #     current_transformation,
            #     open3d.registration.TransformationEstimationPointToPoint(),
            #     open3d.registration.ICPConvergenceCriteria(max_iteration=iter))

            source_down.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                        max_nn=30))
            target_down.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                        max_nn=30))

            # result_icp = open3d.registration.registration_icp(
            #     source_down, target_down, distance_threshold,
            #     current_transformation,
            #     open3d.registration.TransformationEstimationPointToPlane(),
            #     open3d.registration.ICPConvergenceCriteria(max_iteration=iter))

            result_icp = open3d.registration.registration_colored_icp(
                source_down, target_down, voxel_size[scale],
                current_transformation,
                open3d.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=iter))
            current_transformation = result_icp.transformation
            if i == len(max_iter) - 1:
                information_matrix = open3d.registration.get_information_matrix_from_point_clouds(
                    source_down, target_down, voxel_size[scale] * 1.4,
                    result_icp.transformation)
        print(result_icp)
        print(result_icp.transformation)
        return result_icp.transformation, information_matrix


if __name__ == "__main__":
    point_cloud_processing = PointCloudProcessing()
    pcl_left = open3d.io.read_point_cloud("pcl_left.pcd")
    pcl_right = open3d.io.read_point_cloud("pcl_right.pcd")
    point_cloud_processing.process(pcl_left, pcl_right)
    # point_cloud_downsampled = point_cloud_processing.preprocess_point_cloud(pcl_left)
    # open3d.visualization.draw_geometries([point_cloud_downsampled])
    # transformation, information = point_cloud_processing.multiscale_icp(pcl_left, pcl_right, [voxel_size, voxel_size / 2, voxel_size / 4], [50, 30, 14])
