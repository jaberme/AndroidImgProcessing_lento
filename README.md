# Android Image Processing Lento

This Android application is a demonstration of various C++ optimization techniques for image processing tasks. It captures video from the camera and applies different filters and conversions in real time, allowing for a visual comparison of the performance of each method.

## Features

The application processes camera frames, converting them from YUV to RGB, and applies one of the following operations:

*   **RGB Conversion:** Displays the original color image.
*   **Grayscale:** Converts the image to grayscale.
*   **Convolution:** Applies a 3x3 convolution matrix to the image for effects like edge detection or blurring.

For each of these operations, you can select different implementation and optimization strategies:

*   **Java:** A pure Java implementation.
*   **Native (C++):** A basic C++ implementation using the NDK.
*   **pthreads:** A parallel version of the C++ code using POSIX threads.
*   **OpenMP:** A parallel version using the OpenMP framework.
*   **NEON:** An optimized version that leverages ARM NEON intrinsics for accelerated computation.

The application displays the processed video stream and shows the average processing time per frame, allowing for a clear performance comparison between the different methods.

## Building the Project

1.  Clone the repository.
2.  Open the project in Android Studio.
3.  Ensure you have the Android NDK installed through the SDK Manager.
4.  Sync the project with Gradle files.
5.  Build and run the application on a physical device (camera functionality is required).

## Technical Details

*   The native code is written in C++ and can be found in the `app/src/main/cpp` directory.
*   The project uses CMake to build the native `processimg` shared library.
*   Dependencies include `androidx.appcompat` for compatibility with older Android versions.
*   JNI is used to communicate between the Java/Kotlin application code and the C++ native library.
