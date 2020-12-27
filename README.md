## Project: Follow Me - Deep Learning

**Description:**

This is my implementation of Udacity Robotics Software NanoDegree Program Follow-Me Deep Learning project. You are reading writeup!

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

HyperParameter Space

**FCN1:**
Initially I started with FCN1 where parameters as chosen as what "I expect to be around mean"
Even though model is trained in 2 hours (150 seconds per epoch) accuracy was lower than %40. 
And training curve shows that model starts to overfit around 50 epochs for this learning rate, 
as training loss keep decreasing whereas validation loss seem to be constant

FCN1-Training Curve here
FCN1-Hero-Test-1 Here
FCN1-Hero-Test-2 Here

Than I move to FCN2 where I want to investigate the effect of Steps per Epoch parameter. I decreased it to 100 which reduced training time to half (1 hour).
At the end overall accuracy almost same with the FCN1, so I decided to use Steps per Epoch as 100.
This gave me a big training time improvement, allowing me to try different iterations.

FCN2-Training Curve here

Thirdly, I increased Number of Epochs to 100 to see whether or nor model is overfitting and what it's effect on accuracy results.
Training took roughly 2 hours and overall accuracy was %40.7 which is higher than requirement :)
Here is the training curve for FCN3. As it can be seen model starts to overfit around 60 epochs for 0.01 learning rate, which is something we don't want.

FCN3-Training Curve here

After this result, I decided to play with learning rate, while keeping in mind that 60 epochs baseline.
So I decreased Learning rate to 0.004 and epochs to 60. Training took roughly 3 hours and overall accuracy was %42. !Yay!


FCN4-Training Curve here
FCN4-Hero-Test-1 Here
FCN4-Hero-Test-2 Here
FCN4-Hero-Test-3 Here

I was happy with the FCN4 however I want to investigate the effect of Batch size.
For the FCN5 I increased to batch size to 64 which decreased the overall accuracy to %40.4
So I decided to stick with batch size of 32.

Training 

