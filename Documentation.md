
## Models
<br>The images are '28*28', so i thought about using tiny networks (prevent image shrinking)
During ths experiment i used 2 networks:
### Lenet-5 based network
This model we will be having 2 Conv2D layers with 6 and 16 filters each followed by a Max Polling layer of size 2 x 2 Using Eelu activation function.<br> 
We also add a dropout regularization layer with ratio 0.5 to prevent any overfitting.
<br> Together with the dropout I used early stopping technique with a patience interval of 10, which means that after 10 epochs if validation loss is increasing(did not decrease), I stop the training and keep last saved weights 
Finally the model has a softmax layer with the 10 required classes as output.
<br>`NB` : Other modifications to the network:
- A batch normalization layer added after each convolutional layer
- Elu activation function(The original LeNet architecture used TANH activation functions rather than ELU) 
<br>◦ ELU has nonlinearity and linearity at the same time.
<br>◦ No “dying ReLU” problem.
- MaxPool instead of AvgPool 
- padding added to conv1 to prevent the image from being shrinked <br><br>
![Mnist fashion data sample](assets/lenet2.jpg)
#### Network detailed description
A convolution neural net tries to learn the values of filter using backpropagation, which means if a layer has weight matrices, it is called a “learnable” layer. <br>
The number of parameters in a given layer is the count of “learnable” elements for a filter for that layer.<br>
As we can see in the figure, only conv layers and fully connected layers are learnable layers. <br>
The input layer , the pooling layers and the output have 0 learnable parameters.<br>

![Mnist fashion data sample](assets/cnn5params.jpg) <br>
This network contains `60865` parameters almost same as LeNet-5 
### mini_VGG network
![Mnist fashion data sample](assets/minivgg1.jpg)
### mini_VGG Network detailed description
The second model is based on VGG network, the idea behind it is to stack convolutional neural networks with using a bigger number of filters.<br>
As we can see in the figure, this network contains much more parameters then the first model, around `4.8M parameters`., which has made the training slower than the first model.
<br>
In this network, i used 4 convolutional layers, together with max pooling layers and Relu as an activation function. 
This time i used a 512 dense layer .

![Mnist fashion data sample](assets/vggparam.jpg)<br>
In this section, i will try to benchmark between the 4 models we have. As we can see in this table, the model accuracy for all the models is under 90%.
<br> mini_vgg with a batch_size=64 is the winning model with an accuracy 0f `92.2%`

## Benchmarking
I set the training epochs to 150. Used Early stopping techniques to prevent training while model starts overfitting.<br>
As we can see in the table, the training time of each model(experiment), the accuracy  and the number of epochs during which the model was training.
Also you can see diffirent used parameters when training the model.

| model | Accuracy | Training time | Epochs |
| --- | --- | --- | --- |
| CNN5 (lr=0.001, batch_size=64, shuffle=False) | 0.907 | 542| 83|
| CNN5 (lr=0.001, batch_size=128, shuffle=False) | 0.908 | 592| 69 |
| mini_vgg (lr=0.001, batch_size=64, shuffle=False)| 0.922  | 755| 67 |
| mini_vgg (lr=0.001, batch_size=18, shuffle=False)| 0.921| 1152| 40 |

![Mnist fashion data sample](assets/blue.png)  CNN5 (lr=0.001, batch_size=64, shuffle=False) <br>
![Mnist fashion data sample](assets/rose.png) CNN5 (lr=0.001, batch_size=128, shuffle=False)<br>
![Mnist fashion data sample](assets/green.png) mini_vgg (lr=0.001, batch_size=64, shuffle=False)<br>
![Mnist fashion data sample](assets/gray.png)mini_vgg (lr=0.001, batch_size=128, shuffle=False)<br>

### Test loss
![Mnist fashion data sample](assets/Test_Loss.svg)<br>
### Train loss
![Mnist fashion data sample](assets/Train_Loss.svg)<br>
### Accuracy
![Mnist fashion data sample](assets/Accuracy.svg)<br>

As the figures show the green model is the best  (lr=0.001, batch_size=64, shuffle=False) <br>
## Improving the model by more hyper tuning
Due to limited resources/time, i only set few parameters for hyperparameter tuning. <br>
We could improve this by adding more values to 
- learning rate [`1e-3`, `1e-2`, `1e-4`, `1e-1`]
- shuffle [`True`, `False`] : was only set to false
- patience [`10`, `15`, `20`]: earlystopping parameter, set to 10 in our experiment.
- ....


# Bonus Question:
So i wrote a code to make image segmentation by fashion item and i applied trained model on it but i could not find enough time to work on it and process the images:
by using a [segmentation model/object detection model](https://www.kaggle.com/pednoi/training-mask-r-cnn-to-be-a-fashionista-lb-0-07) will help to identify `ROI` regions (region of interest), but it is still not enough to apply the model on it. <br>
![](assets/segmodel.png)
Fashion mnist dataset is a special dataset with black background. <br>
 Generally it is unreasonable to expect the model to work well on data that is different from what it is has been trained on, but If we want to apply this models on real world images, we need to process our images/frames the same way as Fashion Mnist dataset. and even though, it might not generalize on this processed data.  
 <br> In this experiment mini_vgg gave the better results, we could retrain our network with real world images and then apply it video frames/images.
 <br>
 `NB`: `Please do not use my shoes or my trouser any where in the internet without buying a licence :)`