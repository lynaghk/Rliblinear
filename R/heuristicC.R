# Author: Thibault Helleputte
#
# Args:
#	data: a data matrix
#
# Return: the C constant of an SVM, computed following the Joachims' heuristic.
#
heuristicC<-function(data){
	gram=data%*%t(data)
	n=dim(gram)[1]
	kxixi=matrix(ncol=n,nr=1)
	for(i in 1:n){
		kxixi[1,i]=sqrt(gram[i,i])
	}
	m=mean(kxixi)
	C=1/m
	return(C)
}

