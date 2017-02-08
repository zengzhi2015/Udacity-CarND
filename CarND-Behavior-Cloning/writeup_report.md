# **Behavrioal Cloning Project** 

**by Zhi Zeng**

## Abstract

In this project, a model is built to drive a simulated car in a virtual envirenment without human input. In addition, the factors that greately influence the performance of the model are explored. In the project, a convolution neural network in Keras is built to predict steering angles from images. Sufficient data of good driving behavior are collected iteratively according to the performance of the model. Finally, the model can successfully drive around track one without leaving the road.

## I. Intruduction

[Self driving car](https://en.wikipedia.org/wiki/Autonomous_car#cite_note-2) is a vehicle that is capable of sensing its environment and navigating without human input. Among the anticipated benefits of self-driving cars are the potential reduction in traffic collisions that caused by human-driver errors and labor costs.

In spite of these benefits, many technichal  challenges persist. For example, the dilema between the software reliability and the robustness of driverless car in dealing with complex environments. Traditionnal self-driving technique involves manually decompositing the driving task into road or lane marking detection, semantic abstraction, path planning, and control. To enhance the robustness of the system, many human-selected intermediate criteria should be added. However, this would lead to a much complex system and the software reliability would decrease.

Fortunatly, it has been found that this dilema can be solved by using a convolutional neural network (CNN) based controlling system. The research group in Nvidia corporation demonstrated that CNNs are able to learn the entire task of lane and road following without manual decomposition into road or lane marking detection, semantic abstraction, path planning, and control. A small amount of training data from less than a hundred hours of driving was sufﬁcient to train the car to operate in diverse conditions, on highways, local and residential roads in sunny, cloudy, and rainy conditions. The CNN is able to learn meaningful road features from a very sparse training signal (steering alone).[(See this paper)](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

However, the research paper mentioned above did not explore the factors in the technique that most influence the performace of the model. Therefore, to solve this problem, the above mentioned controling system is reproduced and modified for the simulation environment. Key parameters of the model as well as the dataset are varied in a reasonable range and the impact of each factors are explored.

The rest of this report is orgnized as follows. First, the basic structure of the model is introduced. Second, a step by step guide for building and training the model is given. Third, it is explained that how sufficient data of good driving behavior are collected iteratively according to the performance of the model. Finally, those factors that greatly influence the performace of the model are summarized.

It should be noted that my project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [writeup_report.md](./writeup_report.md) summarizing the results
* [data.zip](./data.zip) used for training the model

The car can be driven autonomously around the track by executing 
```
python drive.py
```

## II. Model architecture

The model used for controlling a self driving car using a CNN is clearly stated in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
The structure used in this project is a slightly different from the mentioned work because only a very simple simulation environment is considere here. It can be viewed as a lite version of the model proposed by the Nividia research group.

The CNN model used in this project is also a slightly different from the mentioned work because I want to try transfer learning techniques here. In detail, feature extraction is performed using the convolutionary layers of the [VGG](https://arxiv.org/abs/1409.1556) network. Then, three fully-connected layers are added on top of the feature extraction network for steering angle prediction. Details about the CNN architecture is also shown here.
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
input_1 (InputLayer)             (None, 160, 320, 3)   0                                            
____________________________________________________________________________________________________
block1_conv1 (Convolution2D)     (None, 160, 320, 64)  1792        input_1[0][0]                    
____________________________________________________________________________________________________
block1_conv2 (Convolution2D)     (None, 160, 320, 64)  36928       block1_conv1[0][0]               
____________________________________________________________________________________________________
block1_pool (MaxPooling2D)       (None, 80, 160, 64)   0           block1_conv2[0][0]               
____________________________________________________________________________________________________
block2_conv1 (Convolution2D)     (None, 80, 160, 128)  73856       block1_pool[0][0]                
____________________________________________________________________________________________________
block2_conv2 (Convolution2D)     (None, 80, 160, 128)  147584      block2_conv1[0][0]               
____________________________________________________________________________________________________
block2_pool (MaxPooling2D)       (None, 40, 80, 128)   0           block2_conv2[0][0]               
____________________________________________________________________________________________________
block3_conv1 (Convolution2D)     (None, 40, 80, 256)   295168      block2_pool[0][0]                
____________________________________________________________________________________________________
block3_conv2 (Convolution2D)     (None, 40, 80, 256)   590080      block3_conv1[0][0]               
____________________________________________________________________________________________________
block3_conv3 (Convolution2D)     (None, 40, 80, 256)   590080      block3_conv2[0][0]               
____________________________________________________________________________________________________
block3_pool (MaxPooling2D)       (None, 20, 40, 256)   0           block3_conv3[0][0]               
____________________________________________________________________________________________________
block4_conv1 (Convolution2D)     (None, 20, 40, 512)   1180160     block3_pool[0][0]                
____________________________________________________________________________________________________
block4_conv2 (Convolution2D)     (None, 20, 40, 512)   2359808     block4_conv1[0][0]               
____________________________________________________________________________________________________
block4_conv3 (Convolution2D)     (None, 20, 40, 512)   2359808     block4_conv2[0][0]               
____________________________________________________________________________________________________
block4_pool (MaxPooling2D)       (None, 10, 20, 512)   0           block4_conv3[0][0]               
____________________________________________________________________________________________________
block5_conv1 (Convolution2D)     (None, 10, 20, 512)   2359808     block4_pool[0][0]                
____________________________________________________________________________________________________
block5_conv2 (Convolution2D)     (None, 10, 20, 512)   2359808     block5_conv1[0][0]               
____________________________________________________________________________________________________
block5_conv3 (Convolution2D)     (None, 10, 20, 512)   2359808     block5_conv2[0][0]               
____________________________________________________________________________________________________
block5_pool (MaxPooling2D)       (None, 5, 10, 512)    0           block5_conv3[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 25600)         0           block5_pool[0][0]                
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 2048)          52430848    flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 512)           1049088     dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             513         dense_2[0][0]                    
____________________________________________________________________________________________________
Total params: 68,195,137
Trainable params: 68,195,137
Non-trainable params: 0
```

## III. Procedures

Basic steps of building this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

Details about these steps will be explained in this section.

###1. Collecting the data

Training data was chosen to keep the vehicle driving on the road. Udacity has already provided a set of sample data, but they are far from sufficient. It also suggest to use a combination of center lane driving and recovering from the left and right sides of the road. However, it does not clearly state how much data and how much pattern should be recorded. Actually, the quantity and quality of the recorded samples are quite important for training the model. I will explain this in the next section in detail.

###2. Construction of the convolution neural network

Keras already provides models that can be used for prediction, feature extraction, and fine-tuning. Therefore, there is no need for us to build a full CNN from the bottom. One can extract features with VGG16 (Xception, VGG19, ResNet50, or InceptionV3) over a custom input tensor. In this project, the captured RGB images from the simulator have a size of 320x160 pixels. So, I use the following single line of code to build the feature extractor (code line 67):
```
extractor = VGG16(input_shape=(160,320,3),weights='imagenet', include_top=False)
```
Then, the top layers are built using the following codes. The top layers consist of a RELU layer to introduce nonlinearity and two linear layers so that negative steering commonds can be produced (code lines 73-75 in model.py). The first hidden layer contains 2048 units, the second one contains 512 units, and the output layer has only one unit.
```
features = extractor.output
flat_feature = Flatten()(features)
dense1 = Dense(2048,activation='relu')(flat_feature)
dense2 = Dense(512,activation='linear')(dense1)
predictions = Dense(1,activation='linear')(dense2)
model = Model(input=extractor.input, output=predictions)
```

###3. Training the model

Before traing the model, one should note that the data should be normalized in the model according to the requirement of the input of the VGG model (see code line 54). It should also be note that, the code in model.py uses a Python generator (see code lines 39-56), because the size of the training data is large. 

In addition, in the training stage, only the top layers (which were randomly initialized) are trained, i.e. all convolutional layers are freezed duaring training.
```
for layer in extractor.layers:
   layer.trainable = False
```
The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 59-60). I  randomly shuffled the data set and put 0.05% of the data into a validation set. (A rule of thumb could be to use 80% of the data for training and 20% for validation. However, since my dataset is not large enough and keep a rate of 20% would cause the loss of important patterns, I set a low rate of 0.05% here.) The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86). The batch size for training is 64 and the ideal number of epochs was 3 as evidenced by that the validation accuracy increases when the epoch number is larger than 3.

I do not use the dropout layer to handle the overfitting problem. This is mainly bacause it is difficult to change the dropput rate after the training process in the Keras environment. Instead, I increase the training dataset and use the early stop method.

###4. Testing the model

The final step was to run the simulator to see how well the car was driving around track one. If there were spots where the vehicle fell off the track, to improve the driving behavior in these cases, I would drive the car there by myself and record the correct controlling commond. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

## IV. Dataset construction

It is found that the quality of the dataset is critical to the performance of the model. Although Uddacity has already provide a sample dataset, it is far from sufficient to training a good model. In my test, it is foud that if one use only the sample dataset for training, the car can do nothing but go straight. This is because the dataset is quite unbanlenced. The following figure shows the distribution of the steering commonds:

![Alt text](Fig1)

Therefore, one has to record not only center lane driving, but also recovering driving as well as minor tuning. 

![Alt text](Fig2)
![Alt text](Fig3)
![Alt text](Fig4)

It should be careful that one should not record incorrect behaviors but only the correct ones. For example, as shown in the following figure, if we drive the car in an S shape along a straight road, only the correct stages (marked by the green curves) should be recorded.

![Alt text](Fig5)

Then, the dataset is a balenced one (comparatively speaking). The distribution of the steering commonds would be like this:

![Alt text](Fig6)

Eventhough the dataset is much more balenced, it may not garrentee the performance of the model. There may be spots where the vehicle fell off the track. It shoud be note that, for most cases, this is not because of overfitting, but is due to that the dataset do not cover sufficient modes (or patterns). Therefore, one has to record them many times. Finnaly, I have recorded several rounds of center driving (both clockwise and counter-clockwise along the race) and many recovering cases (especailly near or on the bridges and big turns). My dataset contains 21378 samples in total.

## Critical factors

For the sake of lacking of free time, I only states those foundings here. I will add the detailed exploration process and strict evaluations in the future. 

First, it is found in the tests that the quality of the dataset affects the performance of the model most. If the dataset do not cover sufficient modes, containes too many polluted samples, or it is quite unbalenced, one can by no means obtain a good model for self-driving.

Second, the quantity of the dataset should be large. Otherwise, in the training stage, one should choose a relatively small portion for validation, or importance mode would be loss for training the model.

Third, the dropout technique, mirroring samples, using better feature extractor network, and fine tuning of the feature extractor network do not have profound impact on the performance of the model.


[//]: # (Image References)

[Fig1]: ./Figures/Fig1.PNG "Unbanlenced dataset"
[Fig2]: ./Figures/Fig2.jpg "Center lane driving"
[Fig3]: ./Figures/Fig3.jpg "Recovering driving"
[Fig4]: ./Figures/Fig4.jpg "Minor tuning"
[Fig5]: ./Figures/Fig5.jpg "Correct v.s. incorrect stages"
[Fig6]: ./Figures/Fig6.PNG "Banlenced dataset"