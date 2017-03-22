# reconstructor
A research oriented 3D reconstruction software package.

reconstructor is a simple research-oriented Structure-from-Motion optimizer for image sequences. It takes as input a set of alphabetically ordered images and outputs depth images and absolute poses (up to scale) for each frame. reconstructor is aiming to create a sophisticated reconstruction algorithm which fully utilizes all the information in the input dataset.

**reconstructor is currently under development and will be enhanced and modified to add more features and optimize performance.**

**If you have any questions or if you would like to contribute to this project, please contact Hesam Rabeti at 'hesam.rabeti@tum.de'.**

Quick Start Guide
-------------
Clone this repository:
    
    git clone https://github.com/hesamrabeti/reconstructor.git

Compile:

    cd reconstructor
    cmake .
    make
    
Place dataset in 'reconstructor/data/' (default). You can download some datasets from http://www.votchallenge.net/vot2014/.

Run reconstructor

    ./reconstructor

If you need to resize the images in a folder, you can use the following bash command:

    mkdir resized
    for file in *.jpg; do convert -resize 50% $file resized/$file; done

Technical Details
-------------
Bundle Adjuster optimizes a 6 DOF pose per frame represented by a quaternion rotation and Euclidean translation. Intensity level and depth value consistency is used to align images. A static world is assumed. The static world assumption causes dynamic points of the scene to assume the Gaussian mean depth and intensity level across all frames.

Due to the scale of the optimization problem, very small input images should be used (smaller than 100x100). In the future, Bundle Adjuster will be able to handle larger input images by employing an image pyramid scheme.


Options
-------------
-dataset_directory - Dataset directory

-depth_default_value - Value all depth values are initialized to

-image_step_size - Step size when going through images (Number of images skipped per processed image)

-initial_depth_random_error - Range of random value added to depth values in the initializer

-num_images - Number of images to process

-num_neighbors - Number of neighboring images to use in bundle adjustment

-num_threads - Number of threads used while solving problems

-output_directory - Output directory

-start_image - Image number to start with (Allows for skipping images in the beginning of the dataset)

-suffix - Suffix of image files

-verbose - Print verbose information 

Wishlist
-------------
We would like to implement the following features:

* Use feature-based (ORB/SIFT) methods to initialize the pose and depth of the frames.
* Use an image pyramid scheme to improve convergence characteristics and performance and allow for the use of larger images.
* Obtain a rough depth prior before starting full optimization.
* Detect occlusions.
* Detect dynamic parts of the scene.
* Detect and undo motion blur.
* Handle non-lambertian surfaces.
* Handle rolling shutter cameras.
* Detect scene illumination.
