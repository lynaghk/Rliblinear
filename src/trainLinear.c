#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int nr_fold;
double bias;


double do_cross_validation(void);

void trainLinear(double *W, double *X, double *Y, int *nb_samples, int *nb_dim, int *dim_levels, double *bias, int *type, double *cost, double *epsilon, int *nr_weight, double *weight, int *weight_labels, int *cross, int *verbose);


/**
 * Function: trainLinear
 *
 * Author: Thibault Helleputte, Kevin Lynagh
 *
 */
void trainLinear(double *W, double *X, double *Y, int *nb_samples, int *nb_dim, int *dim_levels, double *bias, int *type, double *cost, double *epsilon, int *nr_weight, double *weight, int *weight_labels, int *cross, int *verbose){

  const char *error_msg;
  int i, j, k;
  i=j=k=0; //counters
  double val;
  int n = *nb_samples; //number of training samples
  int orig_dim = *nb_dim; //number of original dimensions


  //we expand factors so we have a dimension per level, so we need to calculate the dimensionality we're passing to LIBLINEAR
  int p = 0;
  int *offset = Malloc(int, orig_dim);

  for(i=0; i<orig_dim; i++){
    offset[i] = p; //offset i is how many true dimensions come before index i (i.e. if the first column is a factor with three levels, then offset[1] = 3).
    p += dim_levels[i];
  }

  if(*verbose){
    Rprintf("ARGUMENTS SETUP\n");
  }
  param.solver_type = *type;
  param.C = *cost;

  if(!*verbose)
    set_print_string_function(&print_null);

  if(*epsilon <= 0){
    if(param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
      param.eps = 0.01;
    else if(param.solver_type == L2R_L2LOSS_SVC_DUAL || param.solver_type == L2R_L1LOSS_SVC_DUAL || param.solver_type == MCSVM_CS)
      param.eps = 0.1;
    else if(param.solver_type == L1R_L2LOSS_SVC || param.solver_type == L1R_LR)
      param.eps = 0.01;
  }else
    param.eps=*epsilon;

  param.nr_weight = *nr_weight;
  param.weight_label = weight_labels;
  param.weight = weight;

  if(*cross>0){
    flag_cross_validation = 1;
    nr_fold = *cross;
  }else{
    flag_cross_validation = 0;
    nr_fold = 0;
  }

  if(*verbose)
    Rprintf("PROBLEM SETUP\n");
  prob.l = n;
  prob.bias = *bias;

  prob.y = Malloc(int,n);
  prob.x = Malloc(struct feature_node *,n);

  //Allocate space for the features; we need for each training sample one node for:
  //  each feature (since we're getting the info via data frame, we know there will be just orig_dim features per sample, not all p.
  //  one for the bias (if applicable)
  //  one to indicate the end of the feature list for the LIBLINEAR internals
  if(prob.bias >= 0)
    x_space = Malloc(struct feature_node, (orig_dim+1)*n + n);
  else
    x_space = Malloc(struct feature_node, orig_dim*n + n);

  if(*verbose)
    Rprintf("FILL DATA STRUCTURE\n");

  // Fill data stucture
  k=0;
  for(i=0;i<n;i++){
    prob.y[i] = Y[i];
    prob.x[i] = &x_space[k];

    // Fill the sparse feature vector for this sample
    for(j=0; j<orig_dim; j++){
      val = X[(orig_dim*i)+j];
      if(val != -9999){
        if(dim_levels[j] != 1){ //then this is a factor we need to expand
          x_space[k].index = offset[j] + val; //R indexes from 1 as well, so the first factor will have val=1
          x_space[k].value = 1;
        }else{ //this is a numeric column; pass the value
          x_space[k].index = offset[j] + 1; //liblinear indexes from 1
          x_space[k].value = val;
        }
        //print out the representation of this datum passed to liblinear; helpful for catching indexing bugs
        //Rprintf("%d: %lf ", x_space[k].index, x_space[k].value);
        k++;
      }
    }
    //Rprintf("\n");
    if(prob.bias >= 0){
      x_space[k].value = 1;
      x_space[k].index = p+1;
      k++;
    }
    x_space[k].index = -1; //indicate the end of the features for this sample
    k++;
  }

  prob.n = p;
  if(prob.bias >= 0)
    prob.n++; //the bias counts as a feature if we've got it.

  if(*verbose){
    Rprintf("SETUP CHECK\n");
  }

  // SETUP CHECK
  error_msg = NULL;
  error_msg = check_parameter(&prob,&param);

  if(error_msg){
    Rprintf("Error: %s\n",error_msg);
    return;
  }

  if(flag_cross_validation){
    if(*verbose)
      Rprintf("CROSS VAL\n");
    W[0] = do_cross_validation();
  }else{
    if(*verbose)
      Rprintf("TRAIN\n");

    model_ = train(&prob, &param);

    if(*verbose)
      Rprintf("COPY RESULT FOR ");
    if(model_->nr_class==2){
      if(*verbose)
        Rprintf("TWO CLASSES\n");
      for(i=0; i<p; i++)
        W[i] = model_->w[i];
      if(prob.bias >= 0)
        W[i] = model_->w[i];
    }else{
      if(*verbose)
        Rprintf("%d CLASSES\n", model_->nr_class);
      for(i=0; i<model_->nr_class; i++)
        if(prob.bias >= 0)
          for(j=0; j<p+1; j++)
            W[(p+1)*i+j] = model_->w[(p+1)*i+j];
        else
          for(j=0; j<p; j++){
            W[p*i+j]=model_->w[p*i+j];
          }
    }


    free_and_destroy_model(&model_);
  }
  if(*verbose)
    Rprintf("FREE SPACE\n");
  free(prob.y);
  free(prob.x);
  free(x_space);

  return;
}

/**
 * Function: do_cross_validation
 *
 * Author: Thibault Helleputte
 *
 */
double do_cross_validation(void)
{
  int i;
  int total_correct = 0;
  int *target = Malloc(int, prob.l);
  cross_validation(&prob,&param,nr_fold,target);
  for(i=0;i<prob.l;i++)
    if(target[i] == prob.y[i])
      ++total_correct;
  free(target);
  return(1.0*total_correct/prob.l);
}
