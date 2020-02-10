# Fashion Mnist Image classification
The task is to classify Fashion Mnist images to 10 classes which are:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
This is how the dataset looks like: <br>
![Mnist fashion data sample](assets/fashion-mnist-sprite.png)
<br>The images are '28*28', so i thought about using tiny networks (prevent image shrinking)
During ths experiment i used 2 networks:
### Lenet-5 based network
![Mnist fashion data sample](assets/lenet2.jpg)
#### Network detailed description
![Mnist fashion data sample](assets/cnn5.jpg) <br>
`Total number of parameters equal to 58654`

### mini_VGG network
![Mnist fashion data sample](assets/minivgg1.jpg)
### mini_VGG Network detailed description
![Mnist fashion data sample](assets/vgg_mini%20param.jpg)<br>
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