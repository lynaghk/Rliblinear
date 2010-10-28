predict.liblinear = function(object, newx, proba=FALSE, ...){
	
	error=c()
	
	# Nb samples
	n=dim(newx)[1]
	# Nb features
	p=dim(newx)[2]
	
	# Bias
	if(object$bias){
		b=1
	}
	else{
		b=-1
	}
	
	# Return storage preparation
	Y=matrix(nc=n,nr=1,data=0)
	
	# Type 
	if(object$type<0 || object$type>6){
		cat("Invalid model object: Wrong value for 'type'. Must be an integer between 0 and 6 included.\n")
		return(-1)
	}
	
	# Codebook for labels
	cn=c(1:length(object$class_names))
	
	# Proba allowed?
	if(proba && object$type!=0){
		cat("Probabilities only supported for L2-regularized Logistic Regression (liblinear 'type' 0).\n")
		cat("Accordingly, 'proba' is set to FALSE.\n")
		proba=FALSE
	}
	
	#
	# </Arg preparation>
	
	# as.double(t(X)) corresponds to rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time.
	
	ret <- .C(
		"predictLinear",
		as.double(Y),
		as.double(t(newx)),
		as.double(t(object$w)),
		as.integer(proba),
		as.integer(object$nb_class),
		as.integer(p),
		as.integer(n),
		as.double(b),
		as.integer(cn),
		as.integer(object$type)
		)
		
	return(object$class_name[ret[[1]]])

}
