# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/CH_example.png
[image3]: ./output_images/HOG_example.png
[image4]: ./output_images/Feature_Normalization.png
[image5]: ./output_images/sliding_windows.png
[image6]: ./output_images/single_image_feature.png
[image7]: ./output_images/sliding_window.png
[image8]: ./output_images/output_bboxes.png




## I. Feature extraction

This section shows how to perform a histogram of oriented gradients (HOG) feature extraction on a set of images as well as to apply a color transform and append binned color features and histograms of color, to the HOG feature vector. These steps are:

* Reading in all the `vehicle` and `non-vehicle` images
* Compute color features of the image
* Compute color histogram features  
* Compute HOG features and give visualization
* concatenate all features

Following are the details:

---

1.1 Data Exploration

In this section I first reading in both vehicle and non-vehicle image paths provided by the Udacity using the glob.glob() function. 

```
vehicle_files = [name for name in glob.glob('./vehicles/*/*.png')]
nonvehicle_files = [name for name in glob.glob('./non-vehicles/*/*.png')]
```
The number of vehicle images is 8792 and the number of non-vehicle images is 8968. The dataset is quite balenced. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### 1.2 Multiple feature extraction

I first defined a function to compute binned color features. Second, I defined a function to compute color histogram features. 
The code for this step is contained in 6-8 code cells of the IPython notebook located in "./vehicle_detection.ipynb". An example of the color histogram feature is given bellow:

![alt text][image2]

Third I defined a function to return HOG features and visualization. The HOG feature can be extracted using the skimage.feature.hog() function like this:

```
features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)

```

The code for this step is contained in 9-10 code cells of the IPython notebook located in "./vehicle_detection.ipynb". I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Eventually, I use the following parameters:

```
color_space = 'LUV', 
orient = 8, 
pix_per_cell = 8, 
cell_per_block = 1, 
```

I tried various combinations of parameters and choose the 'LUV' color space because this color space seperates the luminance component from the other two color components. The other three parameters are chosen such that the HOG feature is representative and concise as well. I grabbed two images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image3]

Finally, I define a function to extract all features from a list of images. The code for this step is contained in the 11 code cell of the IPython notebook located in "./vehicle_detection.ipynb".

## II. Training Classifier

This section introduce the steps used for training a classifier to distinguish vehicle images and non-vehicle images. These steps are:

* Transform the dataset from the RGB color space to the feature space
* Normalize the features so that they have zero mean and unit variance
* Random shuffle the data and split the data into a training and testing set to avoid overfitting
* Train a linear support vector machine classifier and test its performance

Details are given bellow:

---

### 2.1 Data Preparation

In this section, I first transfor all the images from the RGB color space to the feature space. The code for this step is contained in the 12 code cell of the IPython notebook located in "./vehicle_detection.ipynb".

Second, I do feature normalization, typically to zero mean and unit variance, to avoid individual features or set of features domination the responce of the classifier. The code for this step is contained in the 13 code cell of the IPython notebook located in "./vehicle_detection.ipynb". Here is a comparison between the normalized and the raw features:

![alt text][image4]

### 2.2 Training Classifier

I have first split the dataset into randomized training and testing datasets to avoid the overfitting problem. Then, I tried varias classifiers and found that the support vector machine classifier (SVC) with kernel performed best. It costs 60.27 seconds to train the SVC classifier. The test indicates that the SVC has an accuracy of 0.9927. The code for this step is contained in 15-18 code cells of the IPython notebook located in "./vehicle_detection.ipynb".

## III. Sliding Window Search

This section realize the sliding window search technique. This method involves the following steps:

* Extraction windows of varias sizes
* Resize and transform each extracted window into the feature space
* Normalize and classify each feature
* Record windows containing cars and combine overlapping detections

Following are the details:

---

### 3.1 Sliding window extraction

I first defined a window extraction function which takes takes an image, start and stop positions in both x and y, window size (x and y dimensions), and overlap fraction (for both x and y) as its parameters. Then, I combine all kinds of windows that are useful. The number of small windows is 312. The number of middel windows is 234. The number of large windows is 57. The total number of windows is 603. The following image is a demonstration of sliding windows:

![alt text][image5]

The code for this step is contained in 20-23 code cells of the IPython notebook located in "./vehicle_detection.ipynb".

### 3.2 Classify each window

I first define a function to extract features from a single image window. An example is given to show the difference between the feature of a vehicle and a non-vehicle image:

![alt text][image6]

Then, I define a function which receive an image with the list of windows to be searched and return windows for positive detections. Ultimately, I searched on three scales using luminance HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. An example image is given to show the detection result:

![alt text][image7]

The code for this step is contained in 24-29 code cells of the IPython notebook located in "./vehicle_detection.ipynb".

### 3.3 Combine overlapping detections


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I assumed each blob corresponded to a vehicle.  Thus, I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frame of video:

![alt text][image8]

## IV. Video Implementation

This section combines all the procesures of lane-line detection and car detection together. Corresponding steps includes:

* Warp up all functions regarding car detection
* Warp up all functions regarding car lane-line detection
* Define a buffer to store detected windows in past frames to alliviate the false detection problem

Details are in the following.

---

The video implementation is a little different from the image version in that I have defined a buffer to store detected windows in past frames. These buffered windows are used to smooth the detection and alliviate the false detection problem. The definition of the buffer is given in the 41th code cell of the IPython notebook located in "./vehicle_detection.ipynb". Once a new frame is captured, the detected positive windows, together with previously detected windows, are used to deside the position of the cars.

Here's a [link to my video result](https://youtu.be/RwvUciMPRRQ)

## Discussions

In my implementation of this project, it is found that the quality of the classifier has great influence on the performance the method. Th accuracy of the linear SVC seem not high enough. Therefore I use the kernel version instead. It is also found that the filter (both spacial and temperal) do improve the accuracy of the detection algorithm. I don't think my code is robust enough, because its performace relies on the selection of threshold. In addition, the pipeline might fail in regions with hard shadows. I might improve it if I were going to explore more rubust features.  

