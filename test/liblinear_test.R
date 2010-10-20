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
    type = 0,
    cost = 1,
    bias = TRUE,
    cross = 5,
    verbose = FALSE)

  expect_that( accuracy > 0.8, is_true() )
})
