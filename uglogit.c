// Ugly multinomial logistic regression, with ugly C code.
// Notice that these aren't even C comments!
// This code is based on the libLBFGS example.

#include "stdio.h"
#include "stdlib.h"
#include "getopt.h"
#include "math.h"

#include "lbfgs.h"


// These are hardcoded for the iris data.
// The number of categories includes the arbitrary reference category.
// The number of predictors includes the intercept.
//#define NOBS 150
//#define NCATS 3
//#define NPREDICT 5
//#define NPARAMS ((NCATS - 1) * NPREDICT)


///////////////////////////////////////////////////////////////////////////////
// Training info.

typedef struct tagTRAIN
{
  int nobs;
  int ncats;
  int npredict;
  int nparams;
  double *predict;
  int *respond;
} TRAIN;

int train_init_predict(TRAIN *t, const char *fn_predict)
{
  FILE *f_predict = fopen(fn_predict, "rb");
  if (!f_predict)
  {
    printf("failed to open the prediction data file\n");
    return -1;
  }
  int nrequested = t->nobs * t->npredict;
  t->predict = (double *) malloc(nrequested * sizeof(double));
  int nread = fread(t->predict, sizeof(double), nrequested, f_predict);
  if (nread != nrequested)
  {
    printf("opened but failed to read the right amount of prediction data\n");
    return -1;
  }
  fclose(f_predict);
  return 0;
}

int train_init_respond(TRAIN *t, const char *fn_respond)
{
  FILE *f_respond = fopen(fn_respond, "rb");
  if (!f_respond)
  {
    printf("failed to open the response data file\n");
    return -1;
  }
  int nrequested = t->nobs;
  t->respond = (int *) malloc(nrequested * sizeof(double));
  int nread = fread(t->respond, sizeof(int), nrequested, f_respond);
  if (nread != nrequested)
  {
    printf("opened but failed to read the right amount of response data\n");
    return -1;
  }
  fclose(f_respond);
  return 0;
}

int train_init(TRAIN *t, int n, int d, int k,
    const char *fn_predict, const char *fn_respond)
{
  t->nobs = n;
  t->ncats = d;
  t->npredict = k;
  t->nparams = (d-1) * k;
  t->predict = 0;
  t->respond = 0;
  if (train_init_predict(t, fn_predict) < 0) return -1;
  if (train_init_respond(t, fn_respond) < 0) return -1;
  return 0;
}

int train_destroy(TRAIN *t)
{
  free(t->predict);
  free(t->respond);
  return 0;
}


///////////////////////////////////////////////////////////////////////////////
// More interesting things after here.



// This function is funnily defined so that it can be CUDA'd more easily.
int compute_cat_exps(
    TRAIN *t,
    int i_cat,
    double *cat_exps,
    const lbfgsfloatval_t *x,
    const double *predict
    )
{
  // The last category is the reference category.
  // Its parameter values are all effectively 0.
  if (i_cat == t->ncats-1)
  {
    cat_exps[i_cat] = 1;
    return 0;
  }
  
  // For the other categories, compute an exp of a dot product.
  int i;
  double dot = 0;
  for (i=0; i<t->npredict; ++i)
  {
    dot += x[i_cat * t->npredict + i] * predict[i];
  }
  cat_exps[i_cat] = exp(dot);

  return 0;
}

// This function is funnily defined so that it can be CUDA'd more easily.
int add_to_g(
    TRAIN *t,
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
  if (i_cat == t->ncats-1)
  {
    return 0;
  }

  // For the other categories, accumulate parameter sensitivities.
  int i;
  for (i=0; i<t->npredict; ++i)
  {
    g[i_cat * t->npredict + i] += (cat_exps[i_cat] * predict[i]) / sum_of_exps;
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

  TRAIN *t = (TRAIN *) instance;

  // Initialize the gradient to zero.
  for (i=0; i<t->nparams; ++i)
  {
    g[i] = 0;
  }

  int i_obs, i_cat, i_predict;

  // Eventually put this into some scratchpad storage.
  double *cat_exps = (double *) malloc(t->ncats * sizeof(double));

  // For each observation, do things per category.
  // This structure will be more easily generalized for streaming data.
  for (i_obs=0; i_obs<t->nobs; ++i_obs)
  {

    if ((i_obs+1) % 100000 == 0)
    {
      printf("row %d of %d\n", i_obs+1, t->nobs);
    }

    // Define the predictors and response for this observation.
    double *predict = t->predict + i_obs * t->npredict;
    int respond = t->respond[i_obs];

    // Compute an exp of a dot product for each category.
    for (i_cat=0; i_cat<t->ncats; ++i_cat)
    {
      compute_cat_exps(t, i_cat, cat_exps, x, predict);
    }

    // Compute the sum of exps.
    // This is like the partition function.
    double sum_of_exps = 0;
    for (i_cat=0; i_cat<t->ncats; ++i_cat)
    {
      sum_of_exps += cat_exps[i_cat];
    }

    // Add to the fx which is the negative log likelihood.
    fx -= log(cat_exps[respond] / sum_of_exps);

    // Add to g which is the negative score function,
    // where the score is the gradient of the log likelihood.
    for (i_cat=0; i_cat<t->ncats; ++i_cat)
    {
      add_to_g(t, i_cat, cat_exps, g, sum_of_exps, predict);
    }

    // Adjust g for the observed response.
    if (respond != t->ncats-1)
    {
      for (i_predict=0; i_predict<t->npredict; ++i_predict)
      {
        g[respond * t->npredict + i_predict] -= predict[i_predict];
      }
    }

  }

  //printf("in evaluation function...\n");
  //printf("g:\n");
  //for (i=0; i<t->nparams; ++i)
  //{
    //printf("%f\n", g[i]);
  //}
  printf("fx to be returned from evaluation function: %f\n", fx);

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
  // Initialize some arguments to be read from the command line.
  int n = -1; // number of observations
  int d = -1; // number of categories including the reference category
  int k = -1; // number of predictors including the intercept
  char *p_opt = 0; // predictor filename
  char *r_opt = 0; // response filename
  int verbose = 0; // verbosity

  // Read the command line arguments.
  int c;
  while ( (c = getopt(argc, argv, "vn:d:k:p:r:")) != -1)
  {
    int this_option_optind = optind ? optind : 1;
    switch (c)
    {
      case 'v':
        verbose = 1;
        break;
      case 'n':
        n = atoi(optarg);
        break;
      case 'd':
        d = atoi(optarg);
        break;
      case 'k':
        k = atoi(optarg);
        break;
      case 'p':
        p_opt = optarg;
        break;
      case 'r':
        r_opt = optarg;
        break;
      default:
        printf("?? getopt returned character code 0%o ??\n", c);
    }
  }

  // Check the command line arguments.
  if (n <= 0) {
    printf("n (number of observations) was not specified\n");
    return -1;
  }
  if (d <= 0) {
    printf("d (number of categories including reference) was not specified\n");
    return -1;
  }
  if (k <= 0) {
    printf("k (number of predictors including intercept) was not specified\n");
    return -1;
  }
  if (!p_opt) {
    printf("p (predictor data filename) was not specified\n");
    return -1;
  }
  if (!r_opt) {
    printf("r (response data filename) was not specified\n");
    return -1;
  }

  // Initialize the training data using the command line arguments.
  TRAIN t;
  if (verbose)
  {
    printf("read the training data...\n");
  }
  if (train_init(&t, n, d, k, p_opt, r_opt) < 0) return -1;

  if (verbose)
  {
    printf("allocate lbfgs memory...\n");
  }
  int i, ret = 0;
  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(t.nparams);
  lbfgs_parameter_t param;

  if (x == NULL) {
    printf("failed to allocate a memory block for variables\n");
    return -2;
  }

  // Initialize parameter values to zero.
  if (verbose)
  {
    printf("initialize parameter values to zero...\n");
  }
  for (i=0; i<t.nparams; ++i)
  {
    x[i] = 0;
  }

  /* Initialize the parameters for the L-BFGS optimization. */
  if (verbose)
  {
    printf("initialize parameters for L-BFGS optimization...\n");
  }
  lbfgs_parameter_init(&param);
  /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

  /*
    Start the L-BFGS optimization; this will invoke the callback functions
    evaluate() and progress() when necessary.
   */
  if (verbose)
  {
    printf("starting L-BFGS loop\n");
  }
  ret = lbfgs(t.nparams, x, &fx, evaluate, progress, &t, &param);

  /* Report the result. */
  printf("L-BFGS optimization terminated with status code = %d\n", ret);
  //printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
  printf("  fx = %f\n", fx);

  // Free some memory.
  train_destroy(&t);
  lbfgs_free(x);

  return 0;
}

