# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "undistort_output"
[image2]: ./output_images/unwarped_output.png "unwarped_output"
[image3]: ./output_images/binary_combo_example.png "binary_combo_example"
[image4]: ./output_images/straight_line_example.png "straight_line_example"
[image5]: ./output_images/binary_undistort_example.png "binary_undistort_example"
[image6]: ./output_images/binary_unwrap_example.png "binary_unwrap_example"
[image7]: ./output_images/histogram.png "histogram"
[image8]: ./output_images/sliding_windows.png "sliding_windows"
[image9]: ./output_images/skip_sliding_windows.png "skip_sliding_windows"
[image10]: ./output_images/result.png "result"
[video1]: https://youtu.be/xdR2qeO9TgM "project_output"

## I. Camera Calibration

This section shows the steps to unwarp an image with lens and perspective distortions. These steps are:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Calculate the perspective transform according to the corners of undistorted images and some domain knowllege
* Use cv2.warpPerspective() to warp images to a top-down view

### 1.1 Compute the camera calibration matrix and distortion coefficients

I first prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0). Then, I make a list of calibration images and step through the list and search for chessboard corners use the function cv2.findChessboardCorners(). Finally, I use cv2.calibrateCamera() to calculate the camera matrix, distortion coefficients, rotation, and translation vectors etc.
The code for this step is contained in 1-7 code cells of the IPython notebook located in "./pipeline.ipynb".  

### 1.2. Correct distortion

I define a function to undistort an image. Then, I test the calibration result using chessboard image. The code for this step is contained in 8-9 code cells of the IPython notebook located in "./pipeline.ipynb". The result is given bellow:

![alt text][image1]

### 1.3. Calculate the perspective transform

I try the perspective transform using a chessbord image. Source and destination points are the four corners of the cheeseboard. 
Then, I use cv2.getPerspectiveTransform() to get M, the transform matrix. The undistorted image can be unwarped to a top-down view by using the cv2.warpPerspective() function. I also defined a function to unwarp an undistorted and test the function. The code for this step is contained in 10-13 code cells of the IPython notebook located in "./pipeline.ipynb". The result is given bellow:

![alt text][image2]

## II. Pipeline (single images)

This section shows the steps for calculating the lane on a single image. These steps are:

* Get a distortion-corrected image.
* Used color transforms, gradients or other methods to create a thresholded binary image.
* Performed a perspective transform to the binary image.
* Identified lane-line pixels and fit their positions with a polynomial.
* Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
* Plot the result back down onto the road such that the lane area is identified clearly.

### 2.1 Create binary image

I first create a function to enhance the contrast of the grayscale image which will be used to calculate the edges (or the gradient). The gray image is a combination of the s and v channel of the hsv version of the image. The code for this step is contained in the 14 code cell of the IPython notebook located in "./pipeline.ipynb".

Then, I define four filters to produce thresholded binary images. These filters are the filter on directional gradient, the filter on the magnitude of the gradient, the filter on the direction of the gradient, and the filter on the S channel of the converted HSV image. I combined them up to a single function. The code for this step is contained in 15-19 code cells of the IPython notebook located in "./pipeline.ipynb". I test the function using the following image:

![alt text][image3]

### 2.2 Correct image distortion

I first grayscale an image with straight lines.

![alt text][image4]

Second, I undistort the binary image.

![alt text][image5]

Third, I identify four source points for perspective transform by hand. The source and destination points are in the following manner:

```
src = np.float32([[324,650], [1024,650], [712,450], [606,450]])
dst = np.float32([[color_bin.shape[1]/4,color_bin.shape[0]-1], 
                 [color_bin.shape[1]*3/4,color_bin.shape[0]-1], 
                 [color_bin.shape[1]*3/4,0], 
                 [color_bin.shape[1]/4,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 324, 650      | 320, 720      | 
| 1024, 650     | 960, 720      |
| 712, 450      | 960, 0        |
| 606, 450      | 320, 0        |

I have verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Fourth, I get the transform matrix by using 

```
M = cv2.getPerspectiveTransform(src, dst)
```

Finally, I use cv2.warpPerspective() to warp the image to a top-down view.

![alt text][image6]

### 2.3 Identified lane-line pixels

First, I take a histogram along all the columns in the lower half of the image.

![alt text][image7]

Second, I find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines. The code for this step is contained in the 32 code cell of the IPython notebook located in "./pipeline.ipynb".

Third, I implement the sliding windows method to extract all the pixels of the two lane-lines. For this method, one first select two rectanges roi at the bottom where lies the starting points for the left and right lines. Second, the center of the roi are updated according to the line pixels contained in that roi. Third, the next roi is generated above this roi according to the parameters of the previous roi. After all the windows (roi) are generated, all the line pixels are also gathered. The code for this step is contained in 33-34 code cells of the IPython notebook located in "./pipeline.ipynb".

### 2.4 Fit lane-line pixels positions with a polynomial

Finally, I fit lane-line pixels positions with a polynomial by using np.polyfit(). I test the function using the following image:

![alt text][image8]

### 2.5 Skip the sliding windows step

Once I have the lines identified in an binary image. It's much easier to find line pixels from the next frame of video by considering the line-neighbour area predicted in the previous frame. The code for this step is contained in 37-38 code cells of the IPython notebook located in "./pipeline.ipynb". I test the function using the following image:

![alt text][image9]

### 2.6 Calculated the radius of curvature

To calculate the radius of curvature, I have to first converse pixels in x and y from pixels space to meters. Then, I fit new polynomials to pixels in world space. The code for this step is contained in the 39 code cell of the IPython notebook located in "./pipeline.ipynb".

### 2.7 Calculated the position of the vehicle

I first calculate the position of the left line and right line at the bottom of the image. Then, I do conversions from pixels space to meters and calculate the offset from the center of the two lines. The code for this step is contained in the 40 code cell of the IPython notebook located in "./pipeline.ipynb".

### 2.8 Plot the result back down onto the road

I first warp the overlay to original image space using inverse perspective matrix (Minv).

```
Minv = np.linalg.inv(M)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
```

Then, I combine the result with the original image (the undistorted image). Here is an example of my result:

![alt text][image10]


## III. Pipeline (video)

In this section, I first wrap up all steps discussed above into a single function. The code for this step is contained in the 42 code cell of the IPython notebook located in "./pipeline.ipynb". Then, I read the frames of a videos one by one and apply the function to each frame. Here's a link to my video result.

![alt text][video1]

---

## Discussion

I found in my experiment that it is the quality of the lane-line pixels, the smooth filter, as well as the error checking mechanism that matters a lot. My algorithm fails mostly due to the misdetection of the pixels of the left line when the car reach the shady region. I handle this problem by enhance the contrast of the image and using a low pass filter to smooth the parameters of the predicted lane-lines. My method can be further improved by enhance the performance of both detetion accuracy and smooth filter.

