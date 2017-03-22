// Copyright 2017 Hesam Rabeti
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RECONSTRUCTOR_COST_FUNCTIONS_H_
#define RECONSTRUCTOR_COST_FUNCTIONS_H_

#include <ceres/ceres.h>
#include <ceres/cost_function_to_functor.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace reconstructor {

constexpr double kProjectedPointOutOfBoundsPhotometricError = 0;
constexpr double kProjectedPointOutOfBoundsGeometricError = 0;

template <typename T> inline
void NormalizeQuaternion(T q[4]) {
    const T scale = T(1) / sqrt(q[0] * q[0] +
                                q[1] * q[1] +
                                q[2] * q[2] +
                                q[3] * q[3]);
    q[0] *= scale;
    q[1] *= scale;
    q[2] *= scale;
    q[3] *= scale;
}

double bilinearInterpolation(const cv::Mat& img, const double u, const double v)
{
    int x = (int)u;
    int y = (int)v;

    int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

    double a = u - (double)x;
    double c = v - (double)y;

    return (img.at<double>(y0, x0) * (1.0 - a) + img.at<double>(y0, x1) * a) * (1.0 - c) +
           (img.at<double>(y1, x0) * (1.0 - a) + img.at<double>(y1, x1) * a) * c;
}

// A numerical diff cost function used to look-up the intensity and depth values at the projected pixel locations
struct GrayAndDepthReprojectionError {
public:
    GrayAndDepthReprojectionError(const double source_intensity, const cv::Mat& target_image,
                                  const cv::Mat& target_depth) :
            source_intensity_(source_intensity), target_image_(target_image), target_depth_(target_depth) {}

    bool operator()(const double* target_u,
                    const double* target_v,
                    const double* source_depth,
                    double* residuals) const {
        if(target_u[0] >= 0 && target_v[0] >= 0 &&
           target_u[0] < target_image_.cols && target_v[0] < target_image_.rows) {
            // Projected point is in the bounds of the image.
            residuals[0] = bilinearInterpolation(target_image_, target_u[0], target_v[0]) - source_intensity_;
            residuals[1] = bilinearInterpolation(target_depth_, target_u[0], target_v[0]) - source_depth[0];
        } else {
            // Projected point is out of the bounds of the image. Add a default value to the cost function to make sure
            // removing pixels is not the preferred way to minimize the cost function.
            residuals[0] = kProjectedPointOutOfBoundsPhotometricError;
            residuals[1] = kProjectedPointOutOfBoundsGeometricError;
        }

        return true;
    }

private:
    const double source_intensity_;
    const cv::Mat &target_image_;
    const cv::Mat &target_depth_;
};

// This reprojection error projects the source pixel into the source camera coordinates, then to world coordinate, and
// finally projects the point to target frame. It uses automatic differentiation up until the point it needs to access
// an OpenCV Mat to look-up target intensity and depth values.
struct ReprojectionError {
public:
    ReprojectionError(const int u, const int v, const double source_intensity, const cv::Mat &target_image,
                      const cv::Mat &target_depth)
            : source_u_(u), source_v_(v),
              reprojection_error_calculator_(
                      new ceres::NumericDiffCostFunction<GrayAndDepthReprojectionError, ceres::CENTRAL, 2, 1, 1, 1>(
                              new GrayAndDepthReprojectionError(source_intensity, target_image, target_depth))) {}

    template<typename T>
    bool operator()(const T *const camera_intrinsics,
                    const T *const source_camera_extrinsics, // World to Source camera extrinsics
                    const T *const target_camera_extrinsics, // World to Target camera extrinsics
                    const T *const depth,
                    T *residuals) const {
        T camera_f  = camera_intrinsics[0]; // Focal Length
        T camera_py = camera_intrinsics[1]; // Principal Point px
        T camera_px = camera_intrinsics[2]; // Principal Point py

        // Source camera rotation unit quaternion, inverted.
        T source_camera_q_unit_inv[4] = {source_camera_extrinsics[0], source_camera_extrinsics[1], source_camera_extrinsics[2],
                                 source_camera_extrinsics[3]};
        if(source_camera_q_unit_inv[0] != 0.0 || source_camera_q_unit_inv[1] != 0.0 ||
           source_camera_q_unit_inv[2] != 0.0 || source_camera_q_unit_inv[3] != 0.0 ) {
            NormalizeQuaternion(source_camera_q_unit_inv);
        }
        source_camera_q_unit_inv[1] *= T(-1);
        source_camera_q_unit_inv[2] *= T(-1);
        source_camera_q_unit_inv[3] *= T(-1);

        // Source camera translation.
        T source_camera_t[3] = {source_camera_extrinsics[4], source_camera_extrinsics[5], source_camera_extrinsics[6]};

        // Project from 2D camera plane to 3D source camera coordinate system and then world coordinate.
        T source_depth = depth[0];
        T point[3] = {((T(source_u_) - camera_px) * source_depth) / camera_f,
                      ((T(source_v_) - camera_py) * source_depth) / camera_f,
                      source_depth};
        point[0] += source_camera_t[0];
        point[1] += source_camera_t[1];
        point[2] += source_camera_t[2];
        ceres::UnitQuaternionRotatePoint(source_camera_q_unit_inv, point, point);

        // ---- Point is now in world coordinates

        // Transform into target camera coordinates.
        T target_camera_q[4] = {target_camera_extrinsics[0], target_camera_extrinsics[1],
                                     target_camera_extrinsics[2], target_camera_extrinsics[3]};
        T target_camera_t[3] = {target_camera_extrinsics[4], target_camera_extrinsics[5],
                                     target_camera_extrinsics[6]};
        point[0] -= target_camera_t[0];
        point[1] -= target_camera_t[1];
        point[2] -= target_camera_t[2];
        if(target_camera_q[0] != 0.0 || target_camera_q[1] != 0.0 ||
           target_camera_q[2] != 0.0 || target_camera_q[3] != 0.0 ) {
            ceres::QuaternionRotatePoint(target_camera_q, point, point);
        }

        // ---- Point is now in target camera coordinates

        // Compute final projected point position.
        T target_u = (camera_f * point[0] + point[2] * camera_px) / point[2];
        T target_v = (camera_f * point[1] + point[2] * camera_py) / point[2];

        // Call a special Functor to calculate residual of the reprojected point with the intensity and depth images.
        if(point[2] > T(0)) {
            return reprojection_error_calculator_(&target_u, &target_v, depth, residuals);
        }

        // Point is behind us, ignore this point.
        residuals[0] = T(0);
        residuals[1] = T(0);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction *Create(const int u, const int v, const double source_intensity,
                                       const cv::Mat &target_image, const cv::Mat &target_depth) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 7, 7, 1>(
                new ReprojectionError(u, v, source_intensity, target_image, target_depth)));
    }

private:
    ceres::CostFunctionToFunctor<2, 1, 1, 1> reprojection_error_calculator_;

    const int source_u_; // X Coordinate of pixel in pixel coordinates
    const int source_v_; // Y Coordinate of pixel in pixel coordinates
};

}  // reconstructor

#endif  // RECONSTRUCTOR_COST_FUNCTIONS_H_
