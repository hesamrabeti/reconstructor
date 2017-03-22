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

#include "cost_functions.h"

#include <fstream>
#include <iostream>

#include <dirent.h>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

DEFINE_string   (dataset_directory, "./data/", "Dataset directory");
DEFINE_double   (depth_default_value, 100.0, "Value all depth values are initialized to");
DEFINE_int32    (image_step_size, 10,
    "Step size when going through images  (Number of images skipped per processed image)");
DEFINE_double   (initial_depth_random_error, 0.0, "Range of random value added to depth values in the initializer");
DEFINE_int32    (num_images, 1000, "Number of images to process");
DEFINE_int32    (num_neighbors, 5, "Number of neighboring images to use in bundle adjustment");
DEFINE_int32    (num_threads, 8, "Number of threads used while solving problems");
DEFINE_string   (output_directory, "./output/", "Output directory");
DEFINE_int32    (start_image, 0,
    "Image number to start with (Allows for skipping images in the beginning of the dataset)");
DEFINE_string   (suffix, ".jpg", "Suffix of image files");
DEFINE_bool     (verbose, true, "Print verbose information");

typedef std::array<double, 3> CameraIntrinsics;

bool getImagesInDirectory(std::string directory_path, std::vector<std::string>& output_vector) {
    DIR *dpdf;
    struct dirent *epdf;
    dpdf = opendir(directory_path.c_str());
    if (dpdf != NULL){
        while (epdf = readdir(dpdf)){
            std::string filename(epdf->d_name);

            // If the file's name ends with our desired suffix, add to list of images.
            if(filename.size() >= FLAGS_suffix.size() &&
                    filename.compare(filename.size() - FLAGS_suffix.size(), FLAGS_suffix.size(), FLAGS_suffix) == 0) {
                output_vector.push_back(filename);
            }
        }
    } else {
        std::cout << "Cannot open dataset directory '" << directory_path <<
                  "'. Specify a new directory with -dataset_directory=..." << std::endl;
        exit(1);
    }
    std::sort(output_vector.begin(), output_vector.end());
}

double random(double min, double max)
{
    double d = (double)rand() / RAND_MAX;
    return min + d * (max - min);
}

struct Frame {
    Frame(int image_index, cv::Mat intensity_image) : image_index(image_index), intensity_image(intensity_image) {
        camera_extrinsics = { 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0}; // { Quaternion, Translation }
        depth_image = cv::Mat(intensity_image.size(), CV_64F, FLAGS_depth_default_value);
        // Add some noise to the initial depth image
        for(int v = 0; v < depth_image.rows; ++v) {
            for(int u = 0; u < depth_image.cols; ++u) {
                depth_image.at<double>(v, u) += random(-FLAGS_initial_depth_random_error, FLAGS_initial_depth_random_error);
            }
        }
    }

    Frame(std::array<double, 7> camera_extrinsics, cv::Mat depth_image, cv::Mat intensity_image) :
            camera_extrinsics(camera_extrinsics), depth_image(depth_image), intensity_image(intensity_image){}

    std::array<double, 7> camera_extrinsics;
    cv::Mat               depth_image;
    cv::Mat               intensity_image;
    int                   image_index;
};

// Output the pose and depth of every frame
void WriteOutputs(std::vector<Frame>& frames) {
    std::ofstream poses_file;
    std::string poses_filename = FLAGS_output_directory + "/poses.csv";
    poses_file.open(poses_filename);
    std::cout << "Writing outputs to " << FLAGS_output_directory << "..." << std::endl;
    if(poses_file.is_open()) {
        poses_file << "frame_number,q_w,q_x,q_y,q_z,t_x,t_y,t_z" << std::endl;
        for (size_t i = 0; i < frames.size(); ++i) {
            poses_file << i;
            for(int x = 0; x < frames[i].camera_extrinsics.size(); ++x) {
                poses_file << "," << frames[i].camera_extrinsics[x];
            }
            poses_file << std::endl;
        }
        poses_file.close();
    } else {
        std::cout << "Unable to write to " << poses_filename << "!" << std::endl;
    }

    for(int i = 0; i < frames.size(); ++i) {
        cv::Mat depthOutput;
        frames[i].depth_image.convertTo(depthOutput, CV_8UC1);
        std::string filename = FLAGS_output_directory + "/depth_" + std::to_string(i) + ".png";
        cv::imwrite(filename, depthOutput);
    }
}

void LoadFrames(std::vector<std::string>& images_paths, std::vector<Frame>& frames) {
    // Load all images
    std::cout << "Loading images..." << std::endl;
    for(int index = FLAGS_start_image;
        index < images_paths.size() && index < FLAGS_start_image + (FLAGS_image_step_size * FLAGS_num_images);
        index += FLAGS_image_step_size) {

        // Read Image
        cv::Mat image = cv::imread(FLAGS_dataset_directory + images_paths[index]);

        // Convert to grayscale
        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        // Convert to double for simple interaction with ceres parameters
        image.convertTo(image, CV_64F);
        // Add the image to list of images with origin as the camera extrinsics and default initialized depth
        frames.push_back(Frame(index, image));
    }

}

void SetCameraIntrinsics(CameraIntrinsics& camera_intrinsics, std::vector<Frame>& frames) {
    // Initialize camera intrinsics to a reasonable value
    camera_intrinsics[0] = 1; // Focal length
    // Initialize with centered principal point
    camera_intrinsics[1] = frames[0].intensity_image.cols / 2; // Px
    camera_intrinsics[2] = frames[0].intensity_image.rows / 2; // Py
}

void SetupCeresProblem(CameraIntrinsics& camera_intrinsics, std::vector<Frame>& frames, ceres::Problem& problem) {
    // Add all frames to the problem
    for(int frame_index = 0; frame_index < frames.size(); ++frame_index) {
        // Add neighboring frames to the problem
        for(int neighbor_index = std::max(frame_index - FLAGS_num_neighbors, 0);
            neighbor_index < std::min(frame_index + FLAGS_num_neighbors + 1, static_cast<int>(frames.size()));
            ++neighbor_index) {
            if(frame_index == neighbor_index) {
                // If source and target frames are the same, skip.
                continue;
            }

            // Iterate through all pixels of the image and add them to the problem
            for (int v = 0; v < frames[0].intensity_image.rows; v++) {
                for (int u = 0; u < frames[0].intensity_image.cols; u++) {
                    ceres::CostFunction *costFunction =
                            reconstructor::ReprojectionError::Create(u, v, frames[0].intensity_image.at<double>(v, u),
                                                                       frames[neighbor_index].intensity_image, // target_image
                                                                       frames[neighbor_index].depth_image);    // target_depth

                    // Add a single pixel's residual block
                    double *depth_pixel = &frames[frame_index].depth_image.at<double>(v, u);
                    problem.AddResidualBlock(costFunction, NULL, camera_intrinsics.data(),
                                             frames[frame_index].camera_extrinsics.data(),
                                             frames[neighbor_index].camera_extrinsics.data(),
                                             depth_pixel);
                    problem.SetParameterBlockConstant(camera_intrinsics.data()); // Currently not optimizing intrinsics
                }
            }
        }
    }

}

void SetCeresOptions(ceres::Solver::Options& options) {
    options.minimizer_progress_to_stdout = FLAGS_verbose;
    options.num_threads = FLAGS_num_threads;
    options.num_linear_solver_threads = FLAGS_num_threads;
    options.gradient_tolerance = 1e-18;
    options.function_tolerance = 1e-18;
    options.linear_solver_type = ceres::DENSE_SCHUR;
}

void Init(int* argc, char*** argv) {
    google::ParseCommandLineFlags(argc, argv, true);
    google::InitGoogleLogging(*argv[0]);

    FLAGS_dataset_directory += "/"; // In case the user forgets to add a / at the end of the directory path.

    // Initialize random seed so we have deterministicly random runs.
    srand(12354);
}

int main(int argc, char** argv) {
    Init(&argc, &argv);

    // Get a list of valid images in the dataset_directory
    std::vector<std::string> images_paths;
    getImagesInDirectory(FLAGS_dataset_directory, images_paths);

    if(images_paths.size() == 0) {
        std::cout << "No " << FLAGS_suffix << " files found in '" << FLAGS_dataset_directory <<
        "'. Specify a new directory with -dataset_directory=... or a new suffix with -suffix=..." << std::endl;
        exit(1);
    }


    CameraIntrinsics    camera_intrinsics;
    std::vector<Frame>  frames;

    // Load all frames into memory
    LoadFrames(images_paths, frames);

    // Set a sane initial value for camera parameters
    SetCameraIntrinsics(camera_intrinsics, frames);

    // Set up Ceres Problem
    std::cout << "Setting up Ceres problem..." << std::endl;
    ceres::Problem problem;
    SetupCeresProblem(camera_intrinsics, frames, problem);

    // Set options for Ceres
    ceres::Solver::Options options;
    SetCeresOptions(options);

    // Run Ceres optimization
    ceres::Solver::Summary summary;
    std::cout << "Solving problem..." << std::endl;
    ceres::Solve(options, &problem, &summary);
    if(FLAGS_verbose) {
        std::cout << summary.FullReport() << "\n";
    }

    // Save output
    WriteOutputs(frames);

    return 0;
}
