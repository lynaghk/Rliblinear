liblinear = function(
  data,
  labels,
  type='l2l2_svm_dual',
  cost=1,
  epsilon=0.01,
  bias=TRUE,
  wi=NULL,
  cross=0,
  verbose=FALSE){

  types = list(
    'l2_regression'=0,
    'l2_regression_dual'=7,
    'l1_regression'=6,
    'l2l2_svm'=2,
    'l2l2_svm_dual'=1,
    'l2l1_svm_dual'=3,
    'multiclass'=4,
    'l1l2_svm'=5
    )
  type_err = paste("Wrong value for 'type', must be:\n",
    paste(names(types), collapse="\n"), "\n",
    sep='')

  if(!is.character(type))
    stop(type_err)

  type = types[[type]]
  if(is.null(type))
    stop(type_err)

  #data must be a data frame
  if(class(data) != 'data.frame') data = as.data.frame(data)
  
  # Nb samples
  n=dim(data)[1]
  # Nb features
  orig_dim=dim(data)[2]


  # Epsilon
  if(is.null(epsilon) || epsilon<0){
    # Will use liblinear default value for epsilon
    epsilon = -1
  }

  # Different class penalties?
  y=as.vector(labels)
  yLev=unique(y)
  nbClass=length(yLev)
  yLevC=c(1:nbClass)
  yC=y
  for(i in 1:nbClass){
    ind=which(y==yLev[i])
    yC[ind]=yLevC[i]
  }
  # Default
  defaultWi=rep(1,times=nbClass)
  names(defaultWi)=as.character(yLev)
  nrWi=nbClass
  WiLabels=yLevC

  if(!is.null(wi)){
    if(!is.null(names(wi))){
      if(as.integer(length(intersect(as.character(names(wi)),as.character(yLev))))<length(names(wi))){
        stop("Mismatch between provided names for 'wi' and class labels.\n")
      }
      else{
        Wi=defaultWi
        for(i in 1:length(wi)){
          Wi[as.character(names(wi)[i])]=wi[i]
        }
      }
    }
    else{
      stop("wi has to be a named vector!\n")
    }
  }
  else{
    Wi=defaultWi
  }

  # Cross-validation?
  if(cross<0){
    stop("Cross-validation argument 'cross' cannot be negative!\n")
  }
  else if(cross>n){
    stop("Cross-validation argument 'cross' cannot be larger than the number of samples (",n,").\n",sep="")

  }


  

  #find the number of factor levels for each column; the C program expands these into one dimension per level
  p_levels = sapply(data, function(x){
    l = levels(x)
    return(if (is.null(l)) 1 else length(l))
  })
  p = sum(p_levels)

  #column names with factors expanded (e.g. variable 'a' with three levels will have colnames c('a_level1', 'a_level2', 'a_level3'))
  data_colnames = unlist(sapply(1:orig_dim, function(i){
    orig_colname = colnames(data)[i]
    if(is.null(orig_colname))
      orig_colname = paste('V', i, sep='')
    if(p_levels[i] == 1)
      orig_colname
    else
      paste(orig_colname, levels(data[,i]), sep=' = ')
  }))

  if(bias) data_colnames = c(data_colnames, 'bias')

  # Return storage preparation for result
  if(nbClass==2){
    if(bias){
      W=matrix(nc=p+1,nr=1,data=0)
    }
    else{
      W=matrix(nc=p,nr=1,data=0)
    }
  }
  else if(nbClass>2){
    if(bias)
      W=matrix(nc=(p+1)*nbClass,nr=1,data=0)
    else
      W=matrix(nc=p*nbClass,nr=1,data=0)
  }else
  stop("Wrong number of classes ( < 2 ).\n")



  # as.double(t(data.matrix(X))) corresponds to rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time. Factors are converted to integers
  ret <- .C("trainLinear",
            as.double(W),
            as.double(t(data.matrix(data))),
            as.double(yC),
            as.integer(n), #the number of training data
            as.integer(orig_dim), #the number of columns in the data frame
            as.integer(p_levels), #the number of levels for each dimension (1 for non-factors)
            as.double(if(bias){1}else{-1}),
            as.integer(type),
            as.double(cost),
            as.double(epsilon),
            as.integer(nrWi),
            as.double(Wi),
            as.integer(WiLabels),
            as.integer(cross),
            as.integer(verbose)
            )

  if(cross != 0) #just return the cross validation accuracy reported by liblinear
    return(ret[[1]][1])
  else
    if(nbClass==2) #two class problems get a single weight vector for classification
      w = matrix(nc=dim(W)[2],nr=1,data=ret[[1]])
    else #multiclass problems get a weight vector for each class (one vs. all)
      w = matrix(nc=dim(W)[2]/nbClass,nr=nbClass,data=ret[[1]],byrow=TRUE)

  colnames(w) = data_colnames

  m=list()
  class(m)="liblinear"
  m$type_detail= names(Filter(function(x){x == type}, types))
  m$type=type
  m$w=w
  m$bias=bias
  m$class_names=yLev
  m$nb_class=nbClass
  return(m)
}
