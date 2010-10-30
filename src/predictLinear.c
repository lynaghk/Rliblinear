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

struct feature_node *x;
struct parameter par;
struct model model_;

void predictLinear(double *Y, double *X, double *W, int *proba, int *nb_class, int *nb_dim, int *dim_levels, int *nb_samples, double *bias, int *labels, int *type);


/**
 * Function: predictLinear
 *
 * Author: Thibault Helleputte
 *
 */
void predictLinear(double *Y, double *X, double *W, int *proba, int *nb_class, int *nb_dim, int *dim_levels, int *nb_samples, double *bias, int *labels, int *type){
	
	int i, j, predict_label;
	double *prob_estimates=NULL;
  int orig_dim = *nb_dim;

  //we expand factors so we have a dimension per level, so we need to calculate the dimensionality we're passing to LIBLINEAR
  int p = 0;
  int *offset = Malloc(int, orig_dim);

  for(i=0; i<orig_dim; i++){
    offset[i] = p; //offset i is how many true dimensions come before index i (i.e. if the first column is a factor with three levels, then offset[1] = 3).
    p += dim_levels[i];
  }

	// RECONSTRUCT THE (REQUIRED) PARAMETERS
	par.solver_type=*type;
	
	// RECONSTRUCT THE MODEL
	model_.nr_class=*nb_class;
	model_.nr_feature=p;
	model_.bias=*bias;
	model_.param=par;
	model_.w=W;
	model_.label=labels;
	
  //Allocate space for the features; we can do one sample at a time, and need a node for
  //  each feature (since we're getting the info via data frame, we know there will be just orig_dim features per sample, not all p.
  //  one for the bias (if applicable)
  //  one to indicate the end of the feature list for the LIBLINEAR internals
  if(model_.bias >= 0)
    x = Malloc(struct feature_node, orig_dim+2);
  else
    x = Malloc(struct feature_node, orig_dim+1);

	if(*proba){
		if(model_.param.solver_type!=L2R_LR){
			Rprintf("Probability output is only supported for logistic regression\n");
      return;
    }
		prob_estimates = Malloc(double, *nb_class);
	}

	// PREDICTION PROCESS	
	for(i=0; i<*nb_samples; i++){
		
		for(j=0; j<orig_dim; j++){
			x[j].value = X[(orig_dim*i)+j];
			x[j].index = offset[j]+1; //liblinear indexes from 1
		}

		if(model_.bias>=0){
			x[j].index = p+1;
			x[j].value = model_.bias;
			j++;
		}
		x[j].index = -1;

		if(*proba){
			predict_label = predict_probability(&model_, x, prob_estimates);
			Rprintf("%d",predict_label);
			for(j=0;j<model_.nr_class;j++)
				Rprintf("\t%.8f",prob_estimates[j]);
			Rprintf("\n");
			Y[i]=predict_label;
		}else{
			predict_label = predict(&model_,x);
			Y[i] = predict_label;
		}
	}

	if(*proba)
		free(prob_estimates);
	return;
}

