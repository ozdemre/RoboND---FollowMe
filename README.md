## Project: Follow Me - Deep Learning

**Description:**

This is my implementation of Udacity Robotics Software NanoDegree Program Follow-Me Deep Learning project. You are reading writeup!

For the reviewer here is the files:
1. [Jupyter Notebook Model Training](./code/model_training.ipynb)
2. [HTML version of Model Training](./code/model_training.html)
3. [Model File](./data/weights/model_weights)
4. [Weights File](./data/weights/config_model_weights)
    

**Motivation**

In this project, we built a Fully Convolutional Network(FCN) for performing semantic segmentation on a 
quadcopter simulation for following red hero. FCN's are similar to Convolutional Neural Networks (CNN) where difference comes from adding deconvolution layers (decoder) 
for obtaining spatial information for semantic segmentation. Skip connections can be used to improve spatial information accuracy.

**Network Architecture**

Although many different types of network architecture is possible, I followed my common sense and built my network with 3 encoder layers followed by 1x1 layer and lastly 3 decoder layers.
For details of filters and overall architecture see picture given below





Network picture and Layer sizes here

**Network parameters**

We have a hyperparameter space to tune and optimize for best performance. For that I created a DoE matrix for investigating the effect of parameters. 
Each training model is named as FCN* where architecture is same but hyperparameters differ.
Model parameters and results are given.

HyperParameter Table here

Model Name| Learning Rate | Batch Size | # of Epochs | Steps per Epoch
--- | --- | --- | --- | ---
FCN1 | 0.01 | 32 | 50 | 200
FCN2 | 0.01 | 32 | 50 | 100
FCN3 | 0.01 | 32 | 100 | 100
FCN4 |  0.004 | 32 | 60 | 100
FCN5 |  0.004 | 64 | 60 | 100


**HyperParameter Space**

**FCN1:**
Initially I started with FCN1 where parameters as chosen as what "I expect to be around mean"
Even though model is trained in 2 hours (150 seconds per epoch) accuracy was lower than %40. 
And training curve shows that model starts to overfit around 50 epochs for this learning rate, 
as training loss keep decreasing whereas validation loss seem to be constant


![alt text][image2]
![alt text][image3]
![alt text][image4]

**FCN2:**
Than I move to FCN2 where I want to investigate the effect of Steps per Epoch parameter. I decreased it to 100 which reduced training time to half (1 hour).
At the end overall accuracy almost same with the FCN1, so I decided to use Steps per Epoch as 100.
This gave me a big training time improvement, allowing me to try different iterations.

![alt text][image5]


**FCN3:**
Thirdly, I increased Number of Epochs to 100 to see whether or nor model is overfitting and what it's effect on accuracy results.
Training took roughly 2 hours and overall accuracy was %40.7 which is higher than requirement :)
Here is the training curve for FCN3. As it can be seen model starts to overfit around 60 epochs for 0.01 learning rate, which is something we don't want.

![alt text][image6]

**FCN4:**
After this result, I decided to play with learning rate, while keeping in mind that 60 epochs baseline.
So I decreased Learning rate to 0.004 and epochs to 60. Training took roughly 3 hours and overall accuracy was %42. !Yay!



![alt text][image7]
![alt text][image8]
![alt text][image9]

**FCN5:**
I was happy with the FCN4 however I want to investigate the effect of Batch size.
For the FCN5 I increased to batch size to 64 which decreased the overall accuracy to %40.4
So I decided to stick with batch size of 32.

![alt text][image10]


Training 



[image1]: ./misc/clustering.JPG
[image2]: ./misc/FCN1-Training%20Curve.jpg
[image3]: ./misc/FCN1-Hero-Test-1.jpg
[image4]: ./misc/FCN1-Hero-Test-2.jpg
[image5]: ./misc/FCN2-Training%20Curve.jpg
[image6]: ./misc/FCN3-Training%20Curve.jpg
[image7]: ./misc/FCN4-Training%20Curve.jpg
[image8]: ./misc/FCN4-Hero-Test-1.jpg
[image9]: ./misc/FCN4-Hero-Test-2.jpg
[image10]: ./misc/FCN4-Hero-Test-3.jpg