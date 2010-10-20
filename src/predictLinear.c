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

struct feature_node *x;
struct parameter par;
struct model model_;

void predictLinear(double *Y, double *X, double *W, int *proba, int *nbClass, int *nbDim, int *nbSamples, double *bias, int *labels, int *type);


/**
 * Function: predictLinear
 *
 * Author: Thibault Helleputte
 *
 */
void predictLinear(double *Y, double *X, double *W, int *proba, int *nbClass, int *nbDim, int *nbSamples, double *bias, int *labels, int *type){
	
	int i, j, n, predict_label;
	double *prob_estimates=NULL;
	
	// RECONSTRUCT THE (REQUIRED) PARAMETERS
	par.solver_type=*type;
	
	// RECONSTRUCT THE MODEL
	model_.nr_class=*nbClass;
	model_.nr_feature=*nbDim;
	model_.bias=*bias;
	model_.param=par;
	if(*bias>=0)
		n=*nbDim+1;
	else
		n=*nbDim;
	
	model_.w=W;

	model_.label=labels;
	
	x = (struct feature_node *) malloc(n*sizeof(struct feature_node));
	
	if(*proba){
		if(model_.param.solver_type!=L2R_LR){
			Rprintf("Probability output is only supported for logistic regression\n");
			return;
		}
		prob_estimates = (double *) malloc(*nbClass*sizeof(double));
	}

	// PREDICTION PROCESS	
	for(i=0; i<*nbSamples; i++){
		
		for(j=0; j<*nbDim; j++){
			x[j].value = X[(*nbDim*i)+j];
			x[j].index = j+1;
		}

		if(model_.bias>=0){
			x[j].index = n;
			x[j].value = model_.bias;
			j++;
		}
		x[j].index = -1;

		if(*proba){
			predict_label = predict_probability(&model_,x,prob_estimates);
			Rprintf("%d",predict_label);
			for(j=0;j<model_.nr_class;j++)
				Rprintf("\t%.8f",prob_estimates[j]);
			Rprintf("\n");
			Y[i]=predict_label;
		}
		else{
			predict_label = predict(&model_,x);
			Y[i] = predict_label;
		}
	}
	if(*proba)
		free(prob_estimates);
	return;
}

