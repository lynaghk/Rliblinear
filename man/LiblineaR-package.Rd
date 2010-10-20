\name{LiblineaR-package}
\Rdversion{1.1}
\alias{LiblineaR-package}
\docType{package}
\title{
Linear Predictive Models Based On The Liblinear C/C++ Library.
}
\description{
This R package is a wrapper around the liblinear C/C++ library for machine learning. It allows the estimation of predictive linear models such as L1- or L2-regularized logistic regression, L1- or L2-regularized L2-loss support vector classification, L2-regularized L1-loss support vector classification and multi-class support vector classification. The estimation of the models is particularly fast as compared to other libraries. For more information on liblinear itself, refer to R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. LIBLINEAR: A Library for Large Linear Classification, Journal of Machine Learning Research 9(2008), 1871-1874. Software available at http://www.csie.ntu.edu.tw/~cjlin/liblinear . The two first blocks of the package version indicates which version of liblinear is currently supported. For example: 1.32-14 means that the package supports the version 1.32 of liblinear.
}
\details{
\tabular{ll}{
Package: \tab LiblineaR\cr
Type: \tab Package\cr
Version: \tab 1.51-0\cr
Date: \tab 2010-05-26\cr
License: \tab GPL-2\cr
OS_type: \tab unix\cr
}
The package is a wrapper allowing R users to make use of the liblinear C/C++ library. The two main functions of this package are \code{LiblineaR} itself, and \code{predict.Liblinear}. The first one trains a (generalized) linear model on a given dataset, and the second one uses this model to make prediction (classification) on new data. Multi-class classification is also provided.
}
\author{
Thibault Helleputte\cr
\cr
Maintainer: Thibault Helleputte <thelleputte@gmail.com>
}
\references{
	\itemize{
		\item{This R Package}{
			\url{http://www.thibaulthelleputte.be}
		}
		\item{Liblinear}{
			R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin, \emph{LIBLINEAR: A Library for Large Linear Classification,} Journal of Machine Learning Research 9(2008), 1871-1874. \url{http://www.csie.ntu.edu.tw/~cjlin/liblinear}.
		}
	}
}
\keyword{ package }
\keyword{ classif }
\keyword{ multivariate }
\keyword{ models }
\keyword{ optimize }
\keyword{ classes }
