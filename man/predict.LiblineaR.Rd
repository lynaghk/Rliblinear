\name{predict.LiblineaR}
\Rdversion{1.1}
\alias{predict.LiblineaR}
\title{Classification Predictions With Liblinear}
\description{
	The function takes a model produced by the \code{LiblineaR} function and a data matrix and classifies every row of the matrix as belonging to one of the classes considered by the model.
}
\usage{
	\method{predict}{LiblineaR}(object, newx, proba = FALSE, ...)
}
\arguments{
  \item{object}{
	Object of class \code{"LiblineaR"}, created by \code{LiblineaR}.
}
  \item{newx}{
	An n x p matrix containing the new input data. A vector will be transformed to a n x 1 matrix.
}
  \item{proba}{
	Logical indicating whether class probabilities should be computed and returned. Only possible if the model was fitted with \code{type}=0, i.e. a plain L2-regularized Logistic Regression. Default is \code{FALSE}.
}
  \item{...}{
	Currently not used.
}
}
\details{
	If \code{proba} is \code{TRUE}, a n x k matrix (k number of classes) of the class probabilities is printed.
}
\value{
	A vector of predicted labels. 
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
	Thibault Helleputte\cr
	\email{thelleputte@gmail.com}
}
\note{
	If the data on which the model has been fitted have been centered and/or scaled, it is very important to apply the same process on the \code{newx} data as well, with the scale and center values of the training data.
}

\seealso{
	\code{\link{LiblineaR}}
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