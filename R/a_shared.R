#convert an ordered factor column to numbers in [0, 1].
scale_ordered_factor = function(x){
  l = length(levels(x))
  if(l == 1) return(rep(1.0, length(x)))
  (as.integer(x) - 1) / (length(levels(x)) - 1)
}
  
#convert a data.frame containing factors, logicals, and/or numerics into a numeric matrix.
#This has the same intentions as the built-in 'data.matrix()', but is in some cases (logical matrices) is much faster.
df_to_double = function(df){
  apply(df, 2, function(col){
    #apply converts factor columns into character vectors.
    if(class(col) == "character")
      col = as.integer(factor(col))
    as.double(col)
  })
}
