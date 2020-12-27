## Project: Follow Me - Deep Learning

**Description:**

This is my implementation of Udacity Robotics Software NanoDegree Program Follow-Me Deep Learning project. You are reading writeup!

For the reviewer here is the files:
1. [Jupyter Notebook Model Training](./code/model_training.ipynb)
2. [HTML version of Model Training](./code/model_training.html)
3. [Model File](./data/weights/model_weights)
4. [Weights File](./data/weights/config_model_weights)
5. [Youtube Link for Testing the Model in Simulator](https://www.youtube.com/watch?v=7AhVeR6glBs)
    

**Motivation**

In this project, we built a Fully Convolutional Network(FCN) for performing semantic segmentation on a 
quadcopter simulation for following red hero. FCN's are similar to Convolutional Neural Networks (CNN) where difference comes from adding deconvolution layers (decoder) 
for obtaining spatial information for semantic segmentation. 

**Network Architecture**

Although many different types of network architecture is possible, I followed common sense and built my network with 3 encoder layers followed by 1x1 layer and lastly 3 decoder layers.
For details of filters and overall architecture see picture given below. 1x1 convolution layer is used to preserve the spatial information. For each block same padding and ReLu activation is used.
Encoder layers are used for performing convolutions and highlighting features from raw input. Batch normalization is applied in the encoder_block function.  

```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```
Bilinear upsampling, skip connections and batch normalization is applied in the decoder_block function For both encoder and decoder layers batch normalization is applied.Skip connections can also be used to improve spatial information accuracy.


```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsample_layer = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsample_layer, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concat_layer, filters, 1)
    return output_layer
```

And lastly full network is combined with following function

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder1 = encoder_block(inputs, 64, 2)
    encoder2 = encoder_block(encoder1, 128, 2)
    encoder3 = encoder_block(encoder2, 256, 2)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    one_by_one = conv2d_batchnorm(encoder3, 256, kernel_size = 1, strides = 1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder3 = decoder_block(one_by_one, encoder2, 256)
    decoder2 = decoder_block(decoder3, encoder1, 128)
    decoder1 = decoder_block(decoder2, inputs, 64)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder1)
```

With these properties network is able to perform semantic segmentation by constantly optimizing each layers weights. 
Each encoder block is capturing some features(edges, curves) from it's input.

**Important thing here is FCN does this on by own!**

![alt text][image1]

**Network parameters**

We have a hyperparameter space to tune and optimize for best performance. For that I created a DoE matrix for investigating the effect of parameters. 
Each training model is named as FCN1-5 where architecture is same but hyperparameters differ.
Here is the summary of all model parameters and results.


| Model Name| Learning Rate | Batch Size | # of Epochs | Steps per Epoch | Validation Steps | Loss| Validation Loss | Training Time | Accuracy |
|  ---   | --- | ---   | ---    | ---  | --- | ---    | ---    | ---     | ---  |
| FCN1| 0.01 | 32  | 50  | 200 | 50  | 0.0115 | 0.0246 | 2 hours | %38.6 |
| FCN2| 0.01  | 32  | 50  | 100 | 50 | 0.0131 | 0.0320 | 1 hours | %38.3 |
| FCN3| 0.01 | 32  | 100 | 100 | 50  | 0.0097 | 0.0306 | 2 hours | %40.7  |
| FCN4|  0.004| 32  | 60  | 100 | 50  | 0.0137 | 0.0232 | 3 hours | **%42**   |
| FCN5|  0.004 | 64  | 60  | 100 | 50  | 0.0110 | 0.0289 | 3 hours | %40.4 |


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
This gave me significant training time improvement, allowing me to try different iterations.

![alt text][image5]


**FCN3:**
Thirdly, I increased Number of Epochs to 100 to see whether or nor model is overfitting and what it's effect on accuracy results.
Training took roughly 2 hours and overall accuracy was %40.7 which is higher than requirement :)
Here is the training curve for FCN3. As it can be seen model starts to overfit around 60 epochs for 0.01 learning rate, which is something we don't want.

![alt text][image6]

**FCN4:**
After this result, I decided to play with learning rate, while keeping in mind that 60 epochs baseline.
So I decreased Learning rate to 0.004 and epochs to 60. Training took roughly 3 hours and overall accuracy was %42. **!Yay!**



![alt text][image7]
![alt text][image8]
![alt text][image9]

**FCN5:**
I was happy with the FCN4 however I want to investigate the effect of Batch size.
For the FCN5 I increased to batch size to 64 which decreased the overall accuracy to %40.4
So I decided to stick with batch size of 32.

![alt text][image10]


**Training** 

At the end I was happy with FCN4 hyperparameters:

learning_rate = 0.004

batch_size = 32

num_epochs = 60

steps_per_epoch = 100

validation_steps = 50

workers = 2

With these settings, it took 3 hours to train the model in Udacity Workspace with GPU enabled. Overall accuracy is %42. 

After training is completed, I downloaded the model weight file and test it on the simulator.

Here is [YouTube link](https://www.youtube.com/watch?v=7AhVeR6glBs) where I tested my model within the simulator.

**My Comments and Future Enhancements:**
**Data:** As it is pointed many times in the classes, data quality is very important. Since we started with pre-collected data and have an option of collect 
more from the simulator it is easy to train the model once network is created. However, in real a world implementation, 
I presume this will be more challenging as it requires quite a lot of work to collect thousands of images (maybe with depth info), 
labelling them and doing normalization. I definitely would like to follow these steps outside of the simulator world and take the real-world 
implementation challenges.
In this network we train our model only to detect and perform segmentation on a particular object (red hero). 
Therefore, it has no ability to detect other objects such as cars, trees, dogs, buildings which might be needed for other segmentation requests 
or collision avoidance. For the simulator world this was not needed but for future improvements this is also something to consider.

**Tensorflow Version:** For his project we used Tensorflow v1.2.1. However, there are significant changes on Tensorflow v2 which is currently used one. 
I had struggle to find the documentations for most of the functions we used as the links are outdated. 
As a future study I will try to grasp the changes on v2 and try to implement these steps again.

**Overall:** Even though Neural Networks and Deep Learning was a completely new concept to me, I managed to build a well 
working network from scratch and experiment on the hyperparameter space within a few weeks thanks to well prepared and concise lessons & lab studies. 
In the future I would like to work on this field more, however, I will finish the last lectures of Term-1 and move on to Term-2 for other cool projects!


[image1]: ./misc/FCN_Layout.jpg
[image2]: ./misc/FCN1-Training%20Curve.jpg
[image3]: ./misc/FCN1-Hero-Test-1.jpg
[image4]: ./misc/FCN1-Hero-Test-2.jpg
[image5]: ./misc/FCN2-Training%20Curve.jpg
[image6]: ./misc/FCN3-Training%20Curve.jpg
[image7]: ./misc/FCN4-Training%20Curve.jpg
[image8]: ./misc/FCN4-Hero-Test-1.jpg
[image9]: ./misc/FCN4-Hero-Test-2.jpg
[image10]: ./misc/FCN4-Hero-Test-3.jpg