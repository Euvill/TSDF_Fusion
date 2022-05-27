#pragma once

#include <iostream>
#include <deque>

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>

std::vector<Eigen::Vector4f> get_view_frustum(cv::Mat& depth_im,
                                              const Eigen::Matrix3f& cam_intr, 
                                              const Eigen::Matrix4f& cam_pose);

class TSDFVolume {

public:
    TSDFVolume(const Eigen::Matrix<float, 3, 2> &vol_bnds, float voxel_size);

    bool integrate(const cv::Mat &color_image,
                   const cv::Mat &depth_im,
                   const Eigen::Matrix3f &cam_intr,
                   const Eigen::Matrix4f &cam_pose, 
                   const float obs_weight);

    const int color_const_ = 256 * 256;

    std::vector<std::vector<float>> vox_coords_;

    std::vector<std::vector<std::vector<float>>> tsdf_vol_;
    std::vector<std::vector<std::vector<float>>> weight_vol_;
    std::vector<std::vector<std::vector<float>>> color_vol_;

    long voxel_num_;

private:

    Eigen::Matrix<float, 3, 2> vol_bnds_;
    float voxel_size_;
    float trunc_margin_;

    Eigen::Vector3i vol_dim_;
    Eigen::Vector3f vol_origin_;

    std::vector<Eigen::Vector3f> cam_pts_;

    void flat_color_image(const cv::Mat &src_image, cv::Mat &dst_image);
    void init_3d_tensor(std::vector<std::vector<std::vector<float>>> &volume, const Eigen::Vector3i &dim, float num);
    void init_2d_array(std::vector<std::vector<float>> &volume, const Eigen::Vector2i &dim);
    void vox2world(const Eigen::Vector3f &vol_origin,
                   const float &voxel_size,
                   const Eigen::Vector3i &vol_dim,
                   std::vector<Eigen::Vector3f> &cam_pts);
    void cam2pix(const std::vector<Eigen::Vector3f> cam_pts,
                 const Eigen::Matrix3f &cam_intr,
                 std::vector<Eigen::Vector2i> &pix);

};
