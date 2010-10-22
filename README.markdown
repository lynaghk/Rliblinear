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
Essentially, it is a support vector machine optimized for classes that can be seperated without projecting the data into some fancy-pants kernel space.
For more background, see the [[beginner's guide|http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf]] by Hsu *et al.*

Install the latest version via git:
    git clone http://github.com/lynaghk/Rliblinear/
    R CMD build Rliblinear
    R CMD INSTALL Rliblinear

Ten-fold cross validation:
```r
  require(Rliblinear)
  data(iris)
  liblinear(data = iris[,1:4],
            labels = iris[,5],
            cross = 10,
            cost = 1
            )
  # => 0.95333

  #build a model using two thirds of the iris set
  l = nrow(iris)
  training_indexes = sample(1:l, 2*l / 3)
  model = liblinear(data = iris[training_indexes,1:4],
                    labels = iris[training_indexes,5],
                    cost = 1)

  #compute the accuracy on the remaining third of the iris set
  predictions = predict(model, iris[-training_indexes, 1:4])
  sum( predictions == iris[-training_indexes, 5]) / length(predictions)
  # => 0.98
```




LIBLINEAR
---------
Given training vectors \( \mathbf x_i \in R^n, i = 1,\ldots, l \) and a vector \( \mathbf y \in R^l \) such that \( y_i = \{1, -1\} \), LIBLINEAR builds a weight vector \( \mathbf w \).
This weight vector is a linear predictive model; the decision function is just
\[
  \mathrm{sign}\left(\mathbf w^\mathrm{T} \mathbf x + b\right),
\]
with \(b = 0\) unless `bias = TRUE`

LIBLINEAR solves seven different problems:

*L2-regularized logistic regression*
\[
  \min_{\mathbf w} \quad \frac{1}{2}\mathbf w^{\mathrm{T}}\mathbf w + C \sum_{i=1}^{l} \log\left( 1 + \exp(-y_{i} \mathbf w^{\mathrm T } \mathbf x_{i}) \right)
\]


*L1-regularized logistic regression*
\[
  \min_{\mathbf w} \quad  ||\mathbf w||{}_1 + C \sum_{i=1}^{l} \log\left(1 + \exp(-y_{i} \mathbf w^{\mathrm T}\mathbf x_{i})\right)
\]


*L2-regularized L1-loss support vector classification primal*
\[
  \min_{\mathbf w} \quad  \frac{1}{2}\mathbf w^{\mathrm{T}} \mathbf w + C \sum_{i=1}^{l} \max\left(0, 1 - y_{i} \mathbf{w}^{\mathrm T}\mathbf x_{i} \right)
\]


*L2-regularized L2-loss support vector classification primal*
\[
  \min_{\mathbf w} \quad  \frac{1}{2}\mathbf w^{\mathrm{T}} \mathbf w + C \sum_{i=1}^{l} \max\left(0, 1 - y_{i} \mathbf{w}^{\mathrm T}\mathbf x_{i} \right)^{2}
\]


*L2-regularized L1-loss support vector classification dual*
\[
  \min_{\mathbf \alpha} \quad \frac{1}{2}\mathbf\alpha^{\mathrm{T}} \mathbf{Q} \mathbf\alpha - ||\mathbf\alpha||{}_1 \qquad 0 \le \mathbf\alpha_{i} \le C, \quad i = 1,\ldots, l
\]
where \( \mathbf Q_{ij} = y_i y_j \mathbf x_i^\mathrm{T} \mathbf x_j \).


*L2-regularized L2-loss support vector classification dual*
\[
  \min_{\mathbf \alpha} \quad  \frac{1}{2}\mathbf\alpha^{\mathrm{T}} \mathbf{\overline{Q}} \mathbf\alpha - ||\mathbf\alpha||{}_1 \qquad 0 \le \mathbf\alpha_{i} \le \infty, \quad i = 1,\ldots, l
\]
where \( \mathbf{\overline{Q}} = \mathbf Q + \mathbf D \), with \( D_{ii} = \frac{1}{2C} \).


*L1-regularized L2-loss support vector classification*
\[
  \min_{\mathbf w} \quad ||\mathbf w||{}_1 + C \sum_{i=1}^{l} \max\left(0, 1 - y_{i} \mathbf{w}^{\mathrm T}\mathbf x_{i} \right)^{2}
\]




For more information on liblinear itself, refer to:

R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. 
LIBLINEAR: A Library for Large Linear Classification, 
Journal of Machine Learning Research 9(2008), 1871-1874. 
[[http://www.csie.ntu.edu.tw/~cjlin/liblinear]]




Copyright
---------
All of this software is copyrighted by the list of authors in the DESCRIPTION file of the package and subject to the GNU GENERAL PUBLIC LICENSE, Version 2.

See the file COPYING for details.

The LIBLINEAR C/C++ code is copyright Chih-Chung Chang and Chih-Jen Lin.

The original R wrapper (LiblineaR)  was written by Thibault Helleputte, [[http://www.thibaulthelleputte.be]].
