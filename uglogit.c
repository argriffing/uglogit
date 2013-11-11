// Ugly multinomial logistic regression, with ugly C code.
// Notice that these aren't even C comments!
// This code is based on the libLBFGS example.

#include "stdio.h"
#include "stdlib.h"
#include "lbfgs.h"
#include "math.h"

// These are hardcoded for the iris data.
// The number of categories includes the arbitrary reference category.
// The number of predictors includes the intercept.
#define NOBS 150
#define NCATS 3
#define NPREDICT 5
#define NPARAMS ((NCATS - 1) * NPREDICT)

typedef struct tagTRAIN
{
  double *predict;
  int *respond;
} TRAIN;


// This function is funnily defined so that it can be CUDA'd more easily.
int compute_cat_exps(
    int i_cat,
    double *cat_exps,
    const lbfgsfloatval_t *x,
    const double *predict
    )
{
  // The last category is the reference category.
  // Its parameter values are all effectively 0.
  if (i_cat == NCATS-1)
  {
    cat_exps[i_cat] = 1;
    return 0;
  }
  
  // For the other categories, compute an exp of a dot product.
  int i;
  double dot = 0;
  for (i=0; i<NPREDICT; ++i)
  {
    dot += x[i_cat * NPREDICT + i] * predict[i];
  }
  cat_exps[i_cat] = exp(dot);

  return 0;
}

// This function is funnily defined so that it can be CUDA'd more easily.
int add_to_g(
    int i_cat,
    const double *cat_exps,
    lbfgsfloatval_t *g,
    double sum_of_exps,
    const double *predict
    )
{
  // The last category is the reference category.
  // We do not need sensitivities to changes in its parameters
  // because its parameters are always zero.
  if (i_cat == NCATS-1)
  {
    return 0;
  }

  // For the other categories, accumulate parameter sensitivities.
  int i;
  for (i=0; i<NPREDICT; ++i)
  {
    g[i_cat * NPREDICT + i] += (cat_exps[i_cat] * predict[i]) / sum_of_exps;
  }

  return 0;
}

// This is the evaluation function in the lbfgs loop.
static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
  int i;
  lbfgsfloatval_t fx = 0.0;

  // Initialize the gradient to zero.
  for (i=0; i<NPARAMS; ++i)
  {
    g[i] = 0;
  }

  TRAIN *t = (TRAIN *) instance;

  int i_obs, i_cat, i_predict;

  // Eventually put this into some scratchpad storage.
  double *cat_exps = (double *) malloc(NCATS * sizeof(double));

  // For each observation, do things per category.
  // This structure will be more easily generalized for streaming data.
  for (i_obs=0; i_obs<NOBS; ++i_obs)
  {

    // Define the predictors and response for this observation.
    double *predict = t->predict + i_obs * NPREDICT;
    int respond = t->respond[i_obs];

    // Compute an exp of a dot product for each category.
    for (i_cat=0; i_cat<NCATS; ++i_cat)
    {
      compute_cat_exps(i_cat, cat_exps, x, predict);
    }

    // Compute the sum of exps.
    // This is like the partition function.
    double sum_of_exps = 0;
    for (i_cat=0; i_cat<NCATS; ++i_cat)
    {
      sum_of_exps += cat_exps[i_cat];
    }

    // Add to the fx which is the negative log likelihood.
    fx -= log(cat_exps[respond] / sum_of_exps);

    // Add to g which is the negative score function,
    // where the score is the gradient of the log likelihood.
    for (i_cat=0; i_cat<NCATS; ++i_cat)
    {
      add_to_g(i_cat, cat_exps, g, sum_of_exps, predict);
    }

    // Adjust g for the observed response.
    for (i_predict=0; i_predict<NPREDICT; ++i_predict)
    {
      g[respond * NCATS + i_predict] -= predict[i_predict];
    }

  }

  printf("in evaluation function...\n");
  printf("g:\n");
  for (i=0; i<NPARAMS; ++i)
  {
    printf("%f\n", g[i]);
  }
  printf("fx: %f\n", fx);

  free(cat_exps);

  return fx;
}

// This is an fyi callback function in the lbfgs loop.
static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    //printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}


int main(int argc, char *argv[])
{
  int i, ret = 0;
  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(NPARAMS);
  lbfgs_parameter_t param;

  if (x == NULL) {
    printf("ERROR: Failed to allocate a memory block for variables.\n");
    return 1;
  }

  // Initialize parameter values to zero.
  for (i=0; i<NPARAMS; ++i)
  {
    x[i] = 0;
  }

  // Initialize the training data.
  // TODO free this at the end
  TRAIN t;
  t.predict = (double *) malloc(NOBS * NPREDICT * sizeof(double));
  t.respond = (int *) malloc(NOBS * sizeof(int));
  FILE *f_predict = fopen("iris.predict", "rb");
  fread(t.predict, sizeof(double), NOBS * NPREDICT, f_predict);
  FILE *f_respond = fopen("iris.respond", "rb");
  fread(t.respond, sizeof(int), NOBS, f_respond);

  /* Initialize the parameters for the L-BFGS optimization. */
  lbfgs_parameter_init(&param);
  /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

  /*
    Start the L-BFGS optimization; this will invoke the callback functions
    evaluate() and progress() when necessary.
   */
  ret = lbfgs(NPARAMS, x, &fx, evaluate, progress, &t, &param);

  /* Report the result. */
  printf("L-BFGS optimization terminated with status code = %d\n", ret);
  //printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
  printf("  fx = %f\n", fx);

  lbfgs_free(x);
  return 0;
}

