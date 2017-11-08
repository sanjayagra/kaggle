# kaggle-mnist
This repository contains codes on the Digit Recognizer (MNIST) competition held on Kaggle. These models achieve ~ 99.47% accuracy on the public leaderboard (top 10% solution, may rank even higher if cheaters are ignored).

The basic approach is to build a convolution neural network and improve its performance by using batch normalization, data and test time augmentations. The architecture for the convolution layers is inspired from VGG-16 model that won the imagenet competition in 2013. 

The best score on the public leaderboard comes from ensembling kNN and three CNN models using XGBoost. However, the gain is minor over the single best model.

The current attempts to train a ResNet on this data did not yield any fruitful results. It could be an implementation issue which is being examined.
