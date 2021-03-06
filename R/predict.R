predict.liblinear = function(model, newx, proba=FALSE, ...){

  error=c()

  #coerce a vector into a single-column data frame
  if(!(class(newx) %in% c("matrix", "data.frame")))
    newx = data.frame(newx)
  
  # Nb samples
  n=dim(newx)[1]
  # Nb features
  orig_dim=dim(newx)[2]

  p_levels = sapply(newx, function(x){
    l = levels(x)
    return(if (is.null(l)) 1 else length(l))
  })

  p = sum(p_levels)

  # Return storage preparation
  Y=matrix(nc=n,nr=1,data=0)

  
  # Codebook for labels
  cn=c(1:length(model$class_names))

  # Proba allowed?
  if(proba && !(model$type == 'l2_regression' | model$type == 'l2_regression_dual' )){
    cat("Probabilities only supported for L2-regularized Logistic Regression (liblinear 'type' 0).\n")
    cat("Accordingly, 'proba' is set to FALSE.\n")
    proba=FALSE
  }
  
  #rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time. Factors are converted to integers
  data = t(df_to_double(newx))
  data[is.na(data)] = -9999 #convert NAs to-9999; if trainLinear() sees a value of -9999, it won't pass it along to liblinear.
  ret <- .C(
            "predictLinear",
            as.double(Y),
            as.double(data),
            as.double(t(model$w)),
            as.integer(proba),
            as.integer(model$nb_class),
            as.integer(orig_dim), #the number of columns in the data frame
            as.integer(p_levels), #the number of levels for each dimension (1 for non-factors)
            as.integer(n), #the number of training data
            as.double(if(model$bias){1}else{-1}),
            as.integer(cn),
            as.integer(model$type)
            )

  return(model$class_name[ret[[1]]])

}
