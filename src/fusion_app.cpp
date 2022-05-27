#include <iostream>
#include <deque>
#include <vector>
#include <string>
#include <memory>
#include <omp.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "tsdf_fusion.hpp"

Eigen::MatrixXf loadtxt(const std::string& str) {    
    std::ifstream file_;
    file_.open(str);

    std::string tmp;
    std::vector<std::vector<float>> number;

    Eigen::MatrixXf res;

    while (std::getline(file_, tmp)) {
        
        std::string str_num;
        std::vector<float> num_tmp;

        for (int i = 0; i < tmp.size(); ++i) {

            str_num.push_back(tmp[i]);
            
            if (tmp[i] == ' ' || i == tmp.size() - 1) {
                num_tmp.push_back(std::stof(str_num));
                str_num.clear();
            }
        }

        number.push_back(num_tmp);
    }

    file_.close();

    const int row = number.size();

    if (row == 0) {
        std::cout << "error: no file " << str << std::endl;
        return res;
    }
        
    const int col = number[0].size();
    res.resize(row, col);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            res(i, j) = number[i][j];
        }
    }

    return res;
}

std::string to_seq_index(int num) {
    std::string str = std::to_string(num);
    while (str.size() < 6) {
            str = "0" + str;
    }
    return str;
}

void find_bounds(const std::vector<Eigen::Vector4f>& view_frust_pts, 
                 Eigen::Matrix<float, 3, 2>& vol_bnds) {
                
    for (int i = 0; i < view_frust_pts.size(); ++i) {
        vol_bnds(0, 0) = vol_bnds(0, 0) < view_frust_pts[i](0) ? vol_bnds(0, 0) : view_frust_pts[i](0);
        vol_bnds(0, 1) = vol_bnds(0, 1) > view_frust_pts[i](0) ? vol_bnds(0, 1) : view_frust_pts[i](0);

        vol_bnds(1, 0) = vol_bnds(1, 0) < view_frust_pts[i](1) ? vol_bnds(1, 0) : view_frust_pts[i](1);
        vol_bnds(1, 1) = vol_bnds(1, 1) > view_frust_pts[i](1) ? vol_bnds(1, 1) : view_frust_pts[i](1);

        vol_bnds(2, 0) = vol_bnds(2, 0) < view_frust_pts[i](2) ? vol_bnds(2, 0) : view_frust_pts[i](2);
        vol_bnds(2, 1) = vol_bnds(2, 1) > view_frust_pts[i](2) ? vol_bnds(2, 1) : view_frust_pts[i](2);
    }
}

int main(int argc, char** argv) {

    int N_imgs = 1000;
    Eigen::Matrix3f cam_intr = loadtxt("/home/euvill/Desktop/TSDF_Fusion/data/camera-intrinsics.txt");

    Eigen::Matrix<float, 3, 2> vol_bnds = Eigen::Matrix<float, 3, 2>::Zero();

    std::deque<cv::Mat> depth_im_vec;
    std::deque<cv::Mat> color_im_vec;
    std::deque<Eigen::Matrix4f> cam_pose_vec;

    std::cout << "Estimating voxel volume bounds..." << std::endl;

    for (int i = 0; i < N_imgs; ++i) {

        std::string sqe_str = to_seq_index(i);

        cv::Mat depth_im = cv::imread("/home/euvill/Desktop/TSDF_Fusion/data/frame-" + sqe_str +".depth.png", -1);
        depth_im.convertTo(depth_im, CV_32F); // CV_32F - float data
        depth_im = depth_im / 1000.0; // unit: millimeters
        depth_im_vec.push_back(depth_im);

        cv::Mat color_im = cv::imread("/home/euvill/Desktop/TSDF_Fusion/data/frame-" + sqe_str +".color.jpg", -1);
        cv::Mat dst_color_im;
        cv::cvtColor(color_im, dst_color_im, cv::COLOR_BGR2RGB);
        color_im_vec.push_back(dst_color_im);

        Eigen::Matrix4f cam_pose = loadtxt("/home/euvill/Desktop/TSDF_Fusion/data/frame-" + sqe_str +".pose.txt");
        cam_pose_vec.push_back(cam_pose);

        std::vector<Eigen::Vector4f> view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose);

        find_bounds(view_frust_pts, vol_bnds);
    }

    std::cout << "Initializing voxel volume..." << std::endl;
    std::shared_ptr<TSDFVolume> TSDFVolume_ptr = std::make_shared<TSDFVolume>(vol_bnds, 0.02);
    
    for (int i = 0; i < N_imgs; ++i) {
        
        std::cout << "Fusing frame " << i + 1 << " / " << N_imgs << std::endl;

        cv::Mat depth_im = depth_im_vec[i];
        cv::Mat color_image = color_im_vec[i];
        Eigen::Matrix4f cam_pose = cam_pose_vec[i];

        TSDFVolume_ptr->integrate(color_image, depth_im, cam_intr, cam_pose, 1.0);
    }

    std::cout << "Begin to saving cloud ..." << std::endl;

    long valid_could_point = 0;

    for (long j = 0; j < TSDFVolume_ptr->voxel_num_; ++j) {

        int valid_vox_x = TSDFVolume_ptr->vox_coords_[j][0];
        int valid_vox_y = TSDFVolume_ptr->vox_coords_[j][1];
        int valid_vox_z = TSDFVolume_ptr->vox_coords_[j][2];

        float tsdf = TSDFVolume_ptr->tsdf_vol_[valid_vox_x][valid_vox_y][valid_vox_z];

        if (tsdf > -1 && tsdf < 1) {
            valid_could_point = valid_could_point + 1;
        }
    }

    std::cout << "Valid points' number: " << valid_could_point << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->width  = valid_could_point;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    long index = 0;

    for (long j = 0; j < TSDFVolume_ptr->voxel_num_; ++j) {

        int valid_vox_x = TSDFVolume_ptr->vox_coords_[j][0];
        int valid_vox_y = TSDFVolume_ptr->vox_coords_[j][1];
        int valid_vox_z = TSDFVolume_ptr->vox_coords_[j][2];

        float tsdf = TSDFVolume_ptr->tsdf_vol_[valid_vox_x][valid_vox_y][valid_vox_z];

        if (tsdf > -1 && tsdf < 1) {
            cloud->points[index].x = valid_vox_x;
            cloud->points[index].y = valid_vox_y;
            cloud->points[index].z = valid_vox_z;

            float rgb_vals = TSDFVolume_ptr->color_vol_[valid_vox_x][valid_vox_y][valid_vox_z];
            float colors_b = floor(rgb_vals / TSDFVolume_ptr->color_const_);
            float colors_g = floor((rgb_vals - colors_b * TSDFVolume_ptr->color_const_) / 256);
            float colors_r = rgb_vals - colors_b * TSDFVolume_ptr->color_const_ - colors_g * 256;

            cloud->points[index].r = colors_r;
            cloud->points[index].g = colors_g;
            cloud->points[index].b = colors_b;

            index = index + 1;
        }
    }

    pcl::io::savePCDFileASCII("cloud.pcd", *cloud);

    std::cout << "Visualization." << std::endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Visualizer_Viewer")); 
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud_filtered");		

    viewer->setBackgroundColor(0, 0, 0);	
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,  2, "cloud_filtered");		
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "cloud_filtered");	

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

    return 0;
}