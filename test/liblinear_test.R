source('test_helper.R')

#load up iris dataset for all the tests
data(iris)
x = scale(iris[,1:4], center=TRUE, scale=TRUE)
y = factor(iris[,5])
n = nrow(x)
context('liblinear wrapper')

#not exactly a unit test...
test_that('all problem types are executed properly, with and without bias', {
  types = c(
    'l2_regression',
    'l2_regression_dual',
    'l1_regression',
    'l2l2_svm',
    'l2l2_svm_dual',
    'l2l1_svm_dual',
    'l1l2_svm'
    )
  sapply(types, function(type){
    sapply(c(TRUE, FALSE), function(bias){
      print(paste('type:', type, 'bias:', bias))
      accuracy = liblinear(
        data = x,
        labels = y,
        type = type,
        cost = 1,
        bias = bias,
        cross = 5,
        verbose = FALSE)
      print(accuracy)
      expect_that( accuracy > 0.8, is_true() )
    })
  })
})




test_that('model params are returned correctly', {
  model = liblinear(
    data = x[,1:2],
    labels = y,
    type = 'l2l2_svm_dual',
    cost = 1,
    bias = TRUE,
    verbose = FALSE)

  expect_identical(model$type, 1)
  expect_identical(model$bias, TRUE)
  expect_identical(model$type_detail, 'l2l2_svm_dual')
})


test_that('a model object can be used for prediction', {
  train_i = sample(1:n, 2*n/3)
  train_set = x[train_i,]
  test_set = x[-train_i,]
  model = liblinear(
    data = train_set,
    labels = y[train_i],
    type = 'l2l2_svm_dual',
    cost = 1,
    bias = TRUE,
    verbose = TRUE)

  results = predict(model, test_set)
  accuracy = sum(results == y[-train_i]) / length(results)
  expect_that( accuracy > 0.8, is_true() )
})


test_that('factors are expanded and used', {
  z = as.data.frame(cbind(y, x[,1]))
  z$y = factor(z$y)
  accuracy = liblinear(
    data = z,
    labels = y,
    type = 'l2l2_svm_dual',
    cost = 1,
    cross = 4,
    bias = FALSE,
    verbose = TRUE)

  expect_that( accuracy > 0.99, is_true() )
})


test_that('NAs are handled in factor columns', {
  z = as.data.frame(cbind(y, x[,1:3]))
  z$y = factor(z$y)
  z[sample(1:n, n/3),'y'] = NA
  accuracy = liblinear(
    data = z,
    labels = y,
    type = 'l2l2_svm_dual',
    cost = 1,
    cross = 4,
    bias = FALSE,
    verbose = TRUE)

  expect_that( accuracy > 0.8, is_true() )
})
