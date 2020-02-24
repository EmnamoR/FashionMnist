# Fashion Mnist Image classification
For this experiment, I am going to use the Fashion MNIST dataset that consists of Zalando’s article images which is a set of 28x28 greyscale images of clothes, a drop-in replacement for the MNIST dataset. 
This is how the dataset looks like: <br>
![Mnist fashion data sample](assets/fashion-mnist-sprite.png)
## Getting Started
These instructions will get you a copy of the project up and running on your local machine .
### Prerequisites
- python3.6 installed (app tested on3.6 only)
- Have virtualenv installed on your machine
- create virtual environment 
`conda create --name adver`
or 
`virtualenv adver --python python3`
- activate virtual env
`conda activate adver`
or 
`source path_to_virtualenv/adver/bin/activate`
### Installing
There is 2 releases that one can install <br>
- First relase use crossEntropyLoss + Hyperparameter tuning <br>
```git clone -b v1.0 https://github.com/EmnamoR/AdverFashionMnist.git``` <br>
- Second release uses Triplet Loss/ CrossEntropyLoss + Hyperparameter tuning <br>
```git clone -b v1.2 https://github.com/EmnamoR/AdverFashionMnist.git``` <br>

```pip install -r requirements.txt```
### Run
<br>To run this app 
```
pythton train.py
```
## Documentation
`please find the full documentation of v1.1 release:  `[here](https://github.com/EmnamoR/AdverFashionMnist/blob/master/V1.1-Documentation.md)
`please find the full documentation of v1.3 release(triplet loss release):  `[here](https://github.com/EmnamoR/AdverFashionMnist/blob/master/V1.3-Documentation.md)