source('test_helper.R')

#load up iris dataset for all the tests
data(iris)
x = scale(iris[,1:4], center=TRUE, scale=TRUE)
y = factor(iris[,5])

context('liblinear wrapper')

test_that('iris dataset performance is reasonable', {
  accuracy = liblinear(
    data = x,
    labels = y,
    type = 'l2l2_svm_dual',
    cost = 1,
    bias = TRUE,
    cross = 5,
    verbose = FALSE)

  expect_that( accuracy > 0.8, is_true() )
})

test_that('model params are returned correctly', {
  model = liblinear(
    data = x,
    labels = y,
    type = 'l2l2_svm_dual',
    cost = 1,
    bias = TRUE,
    verbose = FALSE)
  
  expect_identical(model$type, 1)
  expect_identical(model$bias, TRUE)
  expect_identical(model$type_detail, 'l2l2_svm_dual')
  
}
