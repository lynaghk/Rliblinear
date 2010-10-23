                                         _ __  
                                        | '_ \ 
                                        | | | |
      /                             \   |_| |_| 
     /                               \        
     |      +             /          |        
     |       +     +     /           |      
     |                  /   -        |      _____   _  _  _      _  _                           
     |     +    +   +  /  -          |     |  __ \ | |(_)| |    | |(_)                          
     |                /        -     |     | |__) || | _ | |__  | | _  _ __    ___   __ _  _ __ 
     |     +   +     / -   -         |     |  _  / | || || '_ \ | || || '_ \  / _ \ / _` || '__|
     |              /                |     | | \ \ | || || |_) || || || | | ||  __/| (_| || |   
     |    +        /   -    -        |     |_|  \_\|_||_||_.__/ |_||_||_| |_| \___| \__,_||_|   
     |          + /      -           |                                                          
     \           /   -               /                                                          
      \                             /                                                           


This R package is a wrapper around the LIBLINEAR C/C++ library for large linear classification.
LIBLINEAR can handle problems with millions of instances and features.
Essentially, it is a support vector machine optimized for classes that can be seperated without having to project into some fancy-pants kernel space.

Install the latest version via git:

    git clone http://github.com/lynaghk/Rliblinear/
    R CMD build Rliblinear
    R CMD INSTALL Rliblinear

Ten-fold cross validation:

    require(Rliblinear)
    data(iris)
    liblinear(data = iris[,1:4],
              labels = iris[,5],
              cross = 10,
              type = 'l2l2_svm_dual',
              cost = 1
              )
    # => 0.95333
  
    #build a model using two thirds of the iris set
    l = nrow(iris)
    training_indexes = sample(1:l, (2/3)*l)
    model = liblinear(data = iris[training_indexes,1:4],
                      labels = iris[training_indexes,5],
                      type = 'l2l2_svm_dual',
                      cost = 1)

    #compute the accuracy on the remaining third of the iris set
    predictions = predict(model, iris[-training_indexes, 1:4])
    sum( predictions == iris[-training_indexes, 5]) / length(predictions)
    # => 0.98


LIBLINEAR
---------
LIBLINEAR solves seven different problems:

+ L2-regularized logistic regression
+ L1-regularized logistic regression
+ L2-regularized L1-loss support vector classification primal
+ L2-regularized L2-loss support vector classification primal
+ L2-regularized L1-loss support vector classification dual
+ L2-regularized L2-loss support vector classification dual
+ L1-regularized L2-loss support vector classification

For more information on LIBLINEAR itself, refer to [the website](http://www.csie.ntu.edu.tw/~cjlin/liblinear) or see

> R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. 
> LIBLINEAR: A Library for Large Linear Classification, 
> Journal of Machine Learning Research 9(2008), 1871-1874.

For a more practical introduction, see the [beginner's guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) by Hsu *et al.*


Copyright
---------
All of this software is copyrighted by the list of authors in the DESCRIPTION file of the package and subject to the GNU GENERAL PUBLIC LICENSE, Version 2.
See the file COPYING for details.
The LIBLINEAR C/C++ code is copyright Chih-Chung Chang and Chih-Jen Lin.
The original R wrapper (*LiblineaR*)  was written by [Thibault Helleputte](http://www.thibaulthelleputte.be).
