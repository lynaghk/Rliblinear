\name{LiblineaR}
\Rdversion{1.1}
\alias{LiblineaR}
\title{Linear Predictive Models Estimation Based On The Liblinear C/C++ Library.}
\description{
	\code{LiblineaR} allows the estimation of predictive linear models such as L1- or L2-regularized logistic regression, L1- or L2-regularized L2-loss support vector classification, L2-regularized L1-loss support vector classification and multi-class support vector classification. The estimation of the models is particularly fast as compared to other libraries. The implementation is based on the liblinear C/C++ library for machine learning. Multiclass classification is supported. 
}
\usage{
	LiblineaR(data, labels, type=0, cost=1, epsilon = 0.01, bias = TRUE, wi = NULL, cross = 0, verbose = FALSE)
}
\arguments{
  \item{data}{
	a nxp data matrix. Each row stands for an example (sample, point) and each column stands for a dimension (feature, variable)
}
  \item{labels}{
	a response vector for classification tasks with one label for each of the n rows of \code{data}. Can be a 1xn matrix, a simple vector or a factor.
}
  \item{type}{\code{LiblineaR} can produce 7 types of (generalized) linear models, by combining several types of loss functions and regularization schemes. The regularization can be L1 or L2, and the losses can be the regular L2-loss for SVM (hinge loss), L1-loss for SVM, or the logistic loss for logistic regression. The default value for \code{type} is 0. See details below. Valid options are:
		\itemize{
	        \item 0 -- L2-regularized logistic regression
	        \item 1 -- L2-regularized L2-loss support vector classification (dual)
	        \item 2 -- L2-regularized L2-loss support vector classification (primal)
	        \item 3 -- L2-regularized L1-loss support vector classification (dual)
	        \item 4 -- multi-class support vector classification by Crammer and Singer
	        \item 5 -- L1-regularized L2-loss support vector classification
	        \item 6 -- L1-regularized logistic regression
		}
}
  \item{cost}{
	cost of constraints violation (default: 1). Rules the trade-off between regularization and correct classification on \code{data}. It can be seen as the inverse of a regularization constant. See information on the 'C' constant in details below. A usually good baseline heuristics to tune this constant is provided by the \code{heuristicC} function of this package.
}
  \item{epsilon}{
	set tolerance of termination criterion for optimization. Default to 0.01. If \code{NULL}, the liblinear defaults are used, which are:
		\describe{
	        \item{if \code{type} is 0, 2, 5 or 6}{\code{epsilon}=0.01}
	        \item{if \code{type} is 1, 3, or 4}{\code{epsilon}=0.1}
		}
	The meaning of \code{epsilon} is as follows:
		\describe{
	        \item{if \code{type} is 0 or 2:}{\eqn{|f'(w)|_{2} \le \code{epsilon}*\min(pos,neg)/l*|f'(w_{0})|_{2}}{|f'(w)|_2 <= \code{epsilon}*min(pos,neg)/l*|f'(w0)|_2},\cr
	                where f is the primal function and pos/neg are # of positive/negative data.}
	        \item{if \code{type} is 1, 3, or 4:}{Dual maximal violation \eqn{\le}{<=} \code{epsilon}}
	        \item{if \code{type} is 5 or 6:}{\eqn{|f'(w)|_\infty \le \code{epsilon}*\min(pos,neg)/l*|f'(w_{0})|_\infty}{|f'(w)|_inf <= \code{epsilon}*min(pos,neg)/l*|f'(w0)|_inf},\cr
	        		where f is the primal function}
		}
}
  \item{bias}{
	if \code{bias} is \code{TRUE} (default), instances of \code{data} becomes [\code{data}; 1].
}
  \item{wi}{
	a named vector of weights for the different classes, used for asymetric class sizes. Not all factor levels have to be supplied (default weight: 1). All components have to be named according to the corresponding class label.
}
  \item{cross}{
	if an integer value k>0 is specified, a k-fold cross validation on \code{data} is performed to assess the quality of the model via a measure of the accuracy. Note that this metric might not be appropriate if classes are largely unbalanced. Default is 0.
}
  \item{verbose}{
	if \code{TRUE}, information are printed. Default is \code{FALSE}.
}
}
\details{
We report here details for the implementation of liblinear.\cr
\cr
For L2-regularized logistic regression (\code{type}=0), we solve:\cr
\cr
\eqn{\min_{w} \frac{1}{2}w^{T}w + C \sum \log(1 + exp(-y_{i} w^{T}x_{i}))}{min_w w^Tw/2 + C \sum log(1 + exp(-y_i w^Tx_i))}\cr
\cr
For L2-regularized L2-loss SVC dual (\code{type}=1), we solve\cr
\cr
\eqn{\min_{\alpha}  0.5(\alpha^{T} (Q + I/2/C) \alpha) - e^{T} \alpha}{min_alpha  0.5(alpha^T (Q + I/2/C) alpha) - e^T alpha}\cr
s.t.   \eqn{0 \le \alpha_{i}}{0 <= alpha_i},\cr
\cr
For L2-regularized L2-loss SVC (\code{type}=2), we solve\cr
\cr
\eqn{\min_{w} \frac{1}{2}w^{T}w + C \sum \max(0, 1- y_{i} w^{T}x_{i})^{2}}{min_w w^Tw/2 + C sum max(0, 1- y_i w^Tx_i)^2}\cr
\cr
For L2-regularized L1-loss SVC dual (\code{type}=3), we solve\cr
\cr
\eqn{\min_{\alpha}  0.5(\alpha^{T} Q \alpha) - e^{T} \alpha}{min_alpha  0.5(alpha^T Q alpha) - e^T alpha}\cr
s.t.   \eqn{0 \le \alpha_{i} \le C}{0 <= alpha_i <= C},\cr
\cr
For L1-regularized L2-loss SVC (\code{type}=5), we solve\cr
\cr
\eqn{\min_{w} \sum |w_{j}| + C \sum \max(0, 1- y_{i} w^{T}x_{i})^{2}}{min_w sum |w_j| + C sum max(0, 1- y_i w^Tx_i)^2}\cr
\cr
For L1-regularized logistic regression (\code{type}=6), we solve\cr
\cr
\eqn{\min_{w} \sum |w_{j}| + C \sum \log(1 + exp(-y_{i} w^{T}x_{i}))}{min_w sum |w_j| + C sum log(1 + exp(-y_i w^Tx_i))}\cr
\cr
where\cr
\cr
Q is a matrix with \eqn{Q_{ij} = y_{i} y_{j} x_{i}^{T} x_{j}}{Q_ij = y_i y_j x_i^T x_j}.\cr
\cr
If \code{bias} is \code{TRUE}, w becomes [w; w_(n+1)] and x becomes [x; 1].\cr
\cr
The primal-dual relationship implies that \code{type}=1 and \code{type}=2 gives the same model.\cr
\cr
We implement 1-vs-the rest multi-class strategy. In training i vs. non_i, their C parameters are (weight from wi)*C and C, respectively. If there are only two classes, we train only one model. Thus weight1*C vs. weight2*C is used.\cr
\cr
We also implement multi-class SVM by Crammer and Singer (\code{type}=4):\cr
\cr
\eqn{\min_{w_{m}, \xi_{i}}  0.5 \sum_{m} ||w_{m}||^{2} + C \sum_{i} \xi_{i}}{min_{w_m, \xi_i}  0.5 \sum_m ||w_m||^2 + C \sum_i \xi_i}\cr
s.t.  \eqn{w^{T}_{y_{i}} x_{i} - w^{T}_{m} x_{i} \ge e^{m}_{i} - \xi_{i} \forall m,i}{w^T_{y_i} x_i - w^T_m x_i >= e^m_i - \xi_i \forall m,i}\cr
\cr
where\cr
\eqn{e^m_i = 0 \mbox{ if } y_i  = m}{e^m_i = 0 if y_i  = m},\cr
\eqn{e^m_i = 1 \mbox{ if } y_i != m}{e^m_i = 1 if y_i != m},\cr
\cr
Here we solve the dual problem:\cr
\cr
\eqn{\min_{\alpha}  0.5 \sum_{m} ||w_{m}(\alpha)||^{2} + \sum_{i} \sum_{m} e^{m}_{i} \alpha^{m}_{i}}{min_alpha  0.5 sum_m ||w_m(alpha)||^2 + sum_i sum_m e^m_i alpha^m_i}\cr
s.t.  \eqn{\alpha^{m}_{i} \le C^{m}_{i} \forall m,i, \sum_{m} \alpha^{m}_{i}=0 \forall i}{alpha^m_i <= C^m_i forall m,i, sum_m alpha^m_i=0 forall i}\cr
\cr
where \eqn{w_{m}(\alpha) = \sum_{i} \alpha^{m}_{i} x_{i}}{w_m(alpha) = sum_i alpha^m_i x_i},\cr
and\cr
\eqn{C^m_i = C \mbox{ if } m  = y_i}{C^m_i = C if m  = y_i},\cr
\eqn{C^m_i = 0 \mbox{ if } m != y_i}{C^m_i = 0 if m != y_i}.
}
\value{
	If \code{cross}>0, the average accuracy computed over \code{cross} runs of cross-validation is returned.\cr\cr
	Otherwise, an object of class \code{"LiblineaR"} containing the fitted model is returned, including:
	\item{TypeDetail}{A string decsribing the type of model fitted, as determined by \code{type}.}
	\item{Type}{An integer corresponding to \code{type}.}
	\item{W}{A matrix with the model weights. If \code{bias} is TRUE, \code{W} contains p+1 columns, the last being the bias term. The columns are named according to the names of \code{data}, if provided, or \code{"Wx"} where \code{"x"} ranges from 1 to the number of dimensions. The bias term is named \code{"Bias"}.If the number of classes is 2, the matrix only has one row. If the number of classes is k>2, it has k rows. Each row i corresponds then to a linear model discriminating between class i and all the other classes. If there are more than 2 classes, rows are named according to the class i which is opposed to the other classes.}
	\item{Bias}{TRUE or FALSE, according to the value of \code{bias}}
	\item{ClassNames}{A vector containing the class names}
}
\references{
	\itemize{
		\item 
			For more information on liblinear itself, refer to:\cr
			R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.\cr
			\emph{LIBLINEAR: A Library for Large Linear Classification,}\cr
			Journal of Machine Learning Research 9(2008), 1871-1874.\cr
			\url{http://www.csie.ntu.edu.tw/~cjlin/liblinear}
	}
}
\author{
	Thibault Helleputte (based on C/C++-code by Chih-Chung Chang and Chih-Jen Lin)\cr
	\email{thelleputte@gmail.com}
}
\note{
	Classification models usually perform better if each dimension of the data is first centered and scaled.
}

\seealso{
	\code{\link{predict.LiblineaR}},\code{\link{heuristicC}}
}
\examples{
data(iris)
attach(iris)

x=iris[,1:4]
y=factor(iris[,5])
train=sample(1:dim(iris)[1],100)

xTrain=x[train,]
xTest=x[-train,]
yTrain=y[train]
yTest=y[-train]

# Center and scale data
s=scale(xTrain,center=TRUE,scale=TRUE)

# Logistic Regression
t=0

# Tune the cost parameter of a logistic regression via a 10-fold cross-validation
try=c(1000,100,10,1,0.1,0.01,0.001)
res=c()
for(co in try){
	acc=LiblineaR(data=s,labels=yTrain,type=t,cost=co,bias=TRUE,cross=10,verbose=FALSE)
	res=c(res,acc)
	cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
}

# Re-train a model with best C value.
best=which.max(res)
co=try[best]
m=LiblineaR(data=s,labels=yTrain,type=t,cost=co,bias=TRUE,verbose=FALSE)

# Scale the test data
s2=scale(xTest,attr(s,"scaled:center"),attr(s,"scaled:scale"))

# Make prediction
p=predict(m,s2,proba=TRUE)

# Display confusion matrix
res=table(p,yTest)
print(res)

# Compute Balanced Classification Rate
BCR=mean(c(res[1,1]/sum(res[,1]),res[2,2]/sum(res[,2]),res[3,3]/sum(res[,3])))
print(BCR)

}

\keyword{ classif }
\keyword{ multivariate }
\keyword{ models }
\keyword{ optimize }
\keyword{ classes }
