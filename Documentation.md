
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
This network contains `58654` parameters almost same as LeNet-5 (Lenet-5 input is 32*32)

As we can see in the figure there is `60865`
### mini_VGG network
![Mnist fashion data sample](assets/minivgg1.jpg)
### mini_VGG Network detailed description
![Mnist fashion data sample](assets/vggparam.jpg)<br>
`Total number of parameters equal to 2.097.152`


## Benchmarking
| model | Accuracy | Training time |
| --- | --- | --- |
| CNN5 (lr=0.001, batch_size=64, shuffle=False) | 0.907 | 542|
| CNN5 (lr=0.001, batch_size=128, shuffle=False) | 0.908 | 592|
| mini_vgg (lr=0.001, batch_size=64, shuffle=False)| 0.922  | 755|
| mini_vgg (lr=0.001, batch_size=18, shuffle=False)| 0.921| 1152|

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

As the figures show the green model is the best  (lr=0.001, batch_size=64, shuffle=False)