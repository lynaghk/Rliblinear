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

  # Nb samples
  n=dim(data)[1]
  # Nb features
  p=dim(data)[2]

  # Bias
  b = if(bias){1}else{-1}

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

  # Return storage preparation
  if(nbClass==2){
    if(bias){
      W=matrix(nc=p+1,nr=1,data=0)
    }
    else{
      W=matrix(nc=p,nr=1,data=0)
    }
  }
  else if(nbClass>2){
    if(bias){
      W=matrix(nc=(p+1)*nbClass,nr=1,data=0)
    }
    else{
      W=matrix(nc=p*nbClass,nr=1,data=0)
    }
  }
  else{
    stop("Wrong number of classes ( < 2 ).\n")
  }

  #
  # </Arg preparation>

  # as.double(t(X)) corresponds to rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time.
  ret <- .C("trainLinear",
            as.double(W),
            as.double(t(data)),
            as.double(yC),
            as.integer(n),
            as.integer(p),
            as.double(b),
            as.integer(type),
            as.double(cost),
            as.double(epsilon),
            as.integer(nrWi),
            as.double(Wi),
            as.integer(WiLabels),
            as.integer(cross),
            as.integer(verbose)
            )

  if(cross==0){
    if(nbClass==2){
      w=matrix(nc=dim(W)[2],nr=1,data=ret[[1]])
    }
    else{
      w=matrix(nc=dim(W)[2]/nbClass,nr=nbClass,data=ret[[1]],byrow=TRUE)
    }
    if(!is.null(colnames(data))){
      if(bias){
        colnames(w)=c(colnames(data),"Bias")
      }
      else{
        colnames(w)=colnames(data)
      }
    }
    else{
      if(bias){
        colnames(w)=c(paste("W",c(1:dim(data)[2]),sep=""),"Bias")
      }
      else{
        colnames(w)=c(paste("W",c(1:dim(data)[2]),sep=""))
      }
    }

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
  else{
    return(ret[[1]][1])
  }
}
