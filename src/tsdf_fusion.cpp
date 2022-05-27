#include "tsdf_fusion.hpp"

#include <omp.h>
#include <math.h>
#include <chrono>

float find_max_depth(cv::Mat &depth_im) {
    float max = 0;
    for (int i = 0; i < depth_im.rows; ++i) {
        for (int j = 0; j < depth_im.cols; ++j) {

            if(abs(depth_im.at<float>(i, j) - 65.535) <= 1e-3 ) {
                depth_im.at<float>(i, j) = 0.0;
            }
            else{
                max = max < depth_im.at<float>(i, j) ? depth_im.at<float>(i, j) : max;
            }

        }
    }
    return max;
}

std::vector<Eigen::Vector4f> get_view_frustum(cv::Mat& depth_im,
                                              const Eigen::Matrix3f& cam_intr, 
                                              const Eigen::Matrix4f& cam_pose) {
    int im_h = depth_im.rows;
    int im_w = depth_im.cols;

    float max_depth = find_max_depth(depth_im);

    Eigen::Matrix3f cam_intr_inv = cam_intr.inverse();

    std::vector<Eigen::Vector4f> pts_frust_vec;

    Eigen::Vector3f point = max_depth * cam_intr_inv * Eigen::Vector3f(0, 0, 0);
    Eigen::Vector4f pts = Eigen::Vector4f(point(0), point(1), point(2), 1.0);
    pts_frust_vec.push_back(cam_pose * pts);

    point = max_depth * cam_intr_inv * Eigen::Vector3f(0, 0, 1);
    pts = Eigen::Vector4f(point(0), point(1), point(2), 1.0);
    pts_frust_vec.push_back(cam_pose * pts);

    point = max_depth * cam_intr_inv * Eigen::Vector3f(0, im_h, 1);
    pts = Eigen::Vector4f(point(0), point(1), point(2), 1.0);
    pts_frust_vec.push_back(cam_pose * pts);

    point = max_depth * cam_intr_inv * Eigen::Vector3f(im_w, 0, 1);
    pts = Eigen::Vector4f(point(0), point(1), point(2), 1.0);
    pts_frust_vec.push_back(cam_pose * pts);

    point = max_depth * cam_intr_inv * Eigen::Vector3f(im_w, im_h, 1);
    pts = Eigen::Vector4f(point(0), point(1), point(2), 1.0);
    pts_frust_vec.push_back(cam_pose * pts);

    return pts_frust_vec;
}

void TSDFVolume::init_3d_tensor(std::vector<std::vector<std::vector<float>>>& volume, const Eigen::Vector3i& dim, float num) {

    std::vector<std::vector<std::vector<float>>> vec(dim(0), std::vector<std::vector<float>>(
                                                     dim(1), std::vector<float>(
                                                     dim(2), num)));
    volume = vec;
}

void TSDFVolume::init_2d_array(std::vector<std::vector<float>>& volume, const Eigen::Vector2i& dim) {

    std::vector<std::vector<float>> vec(dim(0), std::vector<float>(dim(1), 0.0));

    volume = vec;
}

void TSDFVolume::flat_color_image(const cv::Mat& src_image, cv::Mat& dst_image) {
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            dst_image.at<float>(i, j) =  src_image.at<cv::Vec3f>(i, j)[2] * color_const_ 
                                       + src_image.at<cv::Vec3f>(i, j)[1] * 256 
                                       + src_image.at<cv::Vec3f>(i, j)[0];
        }
    }
}

void TSDFVolume::vox2world(const Eigen::Vector3f &vol_origin, 
                           const float &voxel_size, 
                           const Eigen::Vector3i &vol_dim, 
                           std::vector<Eigen::Vector3f> &cam_pts) {

    long size_coor = vol_dim(0) * vol_dim(1) * vol_dim(2);

#pragma omp parallel
{
#pragma omp for
    for (int i = 0; i < size_coor; ++i) {
        
        Eigen::Vector3f tmp;
        
        tmp(0) = vol_origin(0) + (voxel_size * vox_coords_[i][0]);
        tmp(1) = vol_origin(1) + (voxel_size * vox_coords_[i][1]);
        tmp(2) = vol_origin(2) + (voxel_size * vox_coords_[i][2]);

        cam_pts[i] = tmp;
    }
}
}

void TSDFVolume::cam2pix(const std::vector<Eigen::Vector3f> cam_pts, 
                         const Eigen::Matrix3f& cam_intr,
                         std::vector<Eigen::Vector2i>& pix) {
#pragma omp parallel
{
#pragma omp for
    for (int i = 0; i < cam_pts.size(); ++i) {
        Eigen::Vector3f tmp = cam_intr * cam_pts[i];
        pix[i] = Eigen::Vector2i(round(tmp(0)/tmp(2)), round(tmp(1)/tmp(2)));
    }
}
}


TSDFVolume::TSDFVolume(const Eigen::Matrix<float, 3, 2> &vol_bnds, float voxel_size) {
    // Define voxel volume parameters
    vol_bnds_   = vol_bnds;
    voxel_size_ = voxel_size;
    trunc_margin_ = 5 * voxel_size_;

    // Adjust volume bounds and ensure C-order contiguous
    Eigen::Vector3f tmp = (vol_bnds_.block<3, 1>(0, 1) - vol_bnds_.block<3, 1>(0, 0)) / voxel_size_;
    vol_dim_ = Eigen::Vector3i(ceil(tmp(0)), ceil(tmp(1)), ceil(tmp(2)));
    vol_bnds_.block<3, 1>(0, 1) = vol_bnds_.block<3, 1>(0, 0) + vol_dim_.cast<float>() * voxel_size_;
    vol_origin_ = vol_bnds_.block<3, 1>(0, 0);

    voxel_num_ = vol_dim_(0) * vol_dim_(1) * vol_dim_(2);

    std::cout << "Voxel volume size: " 
              << vol_dim_(0) << " x "  // x-coordinate
              << vol_dim_(1) << " x "  // y-coordinate
              << vol_dim_(2) << " - "  // z-coordinate
              << "# points: " 
              << voxel_num_ << std::endl;

    init_3d_tensor(tsdf_vol_,   vol_dim_, 1.0);
    init_3d_tensor(weight_vol_, vol_dim_, 0.0);
    init_3d_tensor(color_vol_,  vol_dim_, 0.0);

    init_2d_array(vox_coords_, Eigen::Vector2i(voxel_num_, 3));

    long index = 0;
    for (int i = 0; i < vol_dim_(0); ++i) {
        for (int j = 0; j < vol_dim_(1); ++j) {
            for (int k = 0; k < vol_dim_(2); ++k) {
                    vox_coords_[index][0] = i;
                    vox_coords_[index][1] = j;
                    vox_coords_[index][2] = k;
                    index = index + 1;
            }
        }
    }

    std::vector<Eigen::Vector3f> vec(voxel_num_);
    cam_pts_ = vec;

    vox2world(vol_origin_, voxel_size_, vol_dim_, cam_pts_);
}

bool TSDFVolume::integrate(const cv::Mat& color_image, 
                           const cv::Mat& depth_im, 
                           const Eigen::Matrix3f& cam_intr, 
                           const Eigen::Matrix4f& cam_pose, 
                           const float obs_weight) {

    int im_h = depth_im.rows;
    int im_w = depth_im.cols;

    Eigen::Matrix4f cam_pose_inv = cam_pose.inverse();

    // Fold RGB color image into a single channel image
    cv::Mat tmp_color;
    color_image.convertTo(tmp_color, CV_32FC3, 1);
    cv::Mat color_im = cv::Mat(im_h, im_w, CV_32FC1);
    flat_color_image(tmp_color, color_im);

    // Convert voxel grid coordinates to pixel coordinates
    std::vector<Eigen::Vector3f> cam_pts_in_world(cam_pts_.size());
#pragma omp parallel
{
#pragma omp for
    for (int i = 0; i < cam_pts_.size(); ++i) {
        Eigen::Vector4f pts_in_world = cam_pose_inv * Eigen::Vector4f(cam_pts_[i](0), cam_pts_[i](1), cam_pts_[i](2), 1.0);
        cam_pts_in_world[i] = Eigen::Vector3f(pts_in_world(0), pts_in_world(1), pts_in_world(2));
    }
}
    std::vector<Eigen::Vector2i> pix(cam_pts_.size());
    cam2pix(cam_pts_in_world, cam_intr, pix);

    std::vector<float> depth_val(cam_pts_.size(), 0.0);
    // Eliminate pixels outside view frustum
    for (int i = 0; i < cam_pts_.size(); ++i) {
        if (pix[i](0) >= 0 && pix[i](0) < im_w && pix[i](1) >= 0 && pix[i](1) < im_h && cam_pts_in_world[i](2) > 0) {
            depth_val[i] = depth_im.at<float>(pix[i](1), pix[i](0));
        }
    }

    // Integrate TSDF & color
    for (int i = 0; i < cam_pts_.size(); ++i) {
        float depth_diff = depth_val[i] - cam_pts_in_world[i](2);
        if (depth_val[i] > 0 && depth_diff >= -trunc_margin_) {
            float min_tmp = depth_diff / trunc_margin_;
            float dist = 1.0 > min_tmp ? min_tmp : 1.0;

            int valid_vox_x = vox_coords_[i][0];
            int valid_vox_y = vox_coords_[i][1];
            int valid_vox_z = vox_coords_[i][2];

            float w_old = weight_vol_[valid_vox_x][valid_vox_y][valid_vox_z];
            float tsdf_vals = tsdf_vol_[valid_vox_x][valid_vox_y][valid_vox_z];

            float w_new = w_old + obs_weight;
            float tsdf_vol_new = (w_old * tsdf_vals + obs_weight * dist) / w_new;
            
            weight_vol_[valid_vox_x][valid_vox_y][valid_vox_z] = w_new;
            tsdf_vol_[valid_vox_x][valid_vox_y][valid_vox_z] = tsdf_vol_new;

            float old_color = color_vol_[valid_vox_x][valid_vox_y][valid_vox_z];
            float old_b = floor(old_color / color_const_);
            float old_g = floor((old_color - old_b * color_const_) / 256);
            float old_r = old_color - old_b * color_const_ - old_g * 256;

            float new_color = color_im.at<float>(pix[i](1), pix[i](0));
            float new_b = floor(new_color / color_const_);
            float new_g = floor((new_color - new_b * color_const_) / 256);
            float new_r = new_color - new_b * color_const_ - new_g * 256;

            float tmp = round((w_old * old_b + obs_weight * new_b) / w_new);
            new_b = abs(255.0 - tmp) > 1e-3 ? tmp : 255.0;
            tmp = round((w_old * old_g + obs_weight * new_g) / w_new);
            new_g = abs(255.0 - tmp) > 1e-3 ? tmp : 255.0;
            tmp = round((w_old * old_r + obs_weight * new_r) / w_new);
            new_r = abs(255.0 - tmp) > 1e-3 ? tmp : 255.0;

            color_vol_[valid_vox_x][valid_vox_y][valid_vox_z] = new_b * color_const_ + new_g * 256 + new_r;
        }
    }
    //auto t1 = std::chrono::steady_clock::now();
    //auto t2 = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

    return true;
}

