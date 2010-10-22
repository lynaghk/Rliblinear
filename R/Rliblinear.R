liblinear = function(
  data,
  labels,
  type=1,
  cost=1,
  epsilon=0.01,
  bias=TRUE,
  wi=NULL,
  cross=0,
  verbose=FALSE){


  types = list(
    0="l2_regression",
    7="l2_regression_dual",
    6="l1_regression",
    2="l2l2_svm",
    1="l2l2_svm_dual",
    3="l2l1_svm_dual",
    4="multiclass",
    5="l1l2_svm"
    )

  # Nb samples
  n=dim(data)[1]
  # Nb features
  p=dim(data)[2]

  # Bias
  if(bias){
    b=1
  }
  else{
    b=-1
  }

  # Type
  if(type<0 || type>6){
    cat("Wrong value for 'type'. Must be an integer between 0 and 6 included.\n")
    return(-1)
  }

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
        cat("Mismatch between provided names for 'wi' and class labels.\n")
        return(-1)
      }
      else{
        Wi=defaultWi
        for(i in 1:length(wi)){
          Wi[as.character(names(wi)[i])]=wi[i]
        }
      }
    }
    else{
      cat("wi has to be a named vector!\n")
      return(-1)
    }
  }
  else{
    Wi=defaultWi
  }

  # Cross-validation?
  if(cross<0){
    cat("Cross-validation argument 'cross' cannot be negative!\n")
    return(-1)
  }
  else if(cross>n){
    cat("Cross-validation argument 'cross' cannot be larger than the number of samples (",n,").\n",sep="")
    return(-1)
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
    cat("Wrong number of classes ( < 2 ).\n")
    return(-1)
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

    types=c("L2-regularized logistic regression (L2R_LR)", "L2-regularized L2-loss support vector classification dual (L2R_L2LOSS_SVC_DUAL)", "L2-regularized L2-loss support vector classification primal (L2R_L2LOSS_SVC)", "L2-regularized L1-loss support vector classification dual (L2R_L1LOSS_SVC_DUAL)", "multi-class support vector classification by Crammer and Singer (MCSVM_CS)", "L1-regularized L2-loss support vector classification (L1R_L2LOSS_SVC)", "L1-regularized logistic regression (L1R_LR)")
    m=list()
    class(m)="liblinear"
    m$TypeDetail=types[type+1]
    m$Type=type
    m$W=w
    m$Bias=bias
    m$ClassNames=yLev
    m$NbClass=nbClass
    return(m)
  }
  else{
    return(ret[[1]][1])
  }
}
