# OCR
This repository implements the OCR branch of the method introduced in the **E2E-MLT - an Unconstrained End-to-End Method for Multi-Language Scene Text**
paper published by Busta *et al* ([paper](https://arxiv.org/abs/1801.09919)). We wanted to see how the OCR branch performs on traffic signs.

### Pre-training
We have noticed that pre-training the network on synthetic data helps improve the overall performance and speeds up the training process.
For this purpose, we used a subset of the synthetic word dataset of **the Visual Geometry Group** of the University of Oxford ([website](http://www.robots.ox.ac.uk/~vgg/research/text/)).
Our subset contains roughly **20K** training and **2.5K** validation images. 
Here are some samples of these words:
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/1_pontifically_58805.jpg)
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/41_GONER_33090.jpg)
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/88_sidelong_70759.jpg)


### Training
As we mentioned above, we want to recognize writings on traffic signs. Thus, we use images collected from test drives to generate
our training data. Such an image can look like this:
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/img.png)
Then we crop the writings and used them to fine-tune our pre-traied network.
E.g. the crops from the last example are
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/road.png)
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/work.png)
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/ahead.png)

**Note that E2E-MLT can learn to recognize multiple languages at the same time. We only trained it for english and german 
characters.**

All the details regarding the hyperparameters, loss function, etc can be found in the paper mentioned at the very beginning.

### Evaluation
After training/fine-tuning the network for 10 epochs on our data, we use it for prediction. For the sake of continuity, we use
the same image, i.e. the same crops, to give a feeling of what the network does. 
The figure below summarizes all the steps we discussed so far.
![alt text](https://github.com/adnenabdessaied/OCR/blob/master/md_images/img_.png)

