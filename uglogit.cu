// Ugly multinomial logistic regression, with ugly C code.
// Notice that these aren't even C comments!
// This code is based on the libLBFGS example.

#include "stdio.h"
#include "stdlib.h"
#include "getopt.h"

#include "lbfgs.h"


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
// CUDA functions.

// Each observation row could be a block,
// and each category could be a thread within the block.
// Or blockIdx.y will be the data row and blockIdx.x
// will allow multiple blocks to cover the categories,
// with up to 32 threads running at a time per x block.
// So this would work best when the number of categories is a multiple of 32.
//
// The __global__ keyword indicates a function that runs on the device
// and is called from host code.
__global__
void compute_nonlast_cat_exps(
    int ncats,
    int npredict,
    double *cat_exps_out,
    const double *params_in,
    const double *predict_in
    )
{
  // We are currently using only one block,
  // so the thread index defines the category.
  // The last category is the reference category,
  // whose parameter values are all zero and must be treated separately,
  // so we do not process it here to avoid branching on the thread index.
  // Its parameter values are all effectively 0.
  // For the other categories, compute an exp of a dot product.
  int i_predict;
  double dot;
  int i_cat = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_cat < ncats)
  {
    dot = 0;
    for (i_predict=0; i_predict<npredict; ++i_predict)
    {
      dot += params_in[i_cat * npredict + i_predict] * predict_in[i_predict];
    }
    cat_exps_out[i_cat] = exp(dot);
  }
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

  // Allocate cuda memory.
  double *d_cat_exps, *d_params, *d_predict_all;
  cudaMalloc((void **) &d_cat_exps, t->ncats * sizeof(double));
  cudaMalloc((void **) &d_params, t->nparams * sizeof(double));
  cudaMalloc((void **) &d_predict_all, t->nobs * t->npredict * sizeof(double));

  // Copy all parameter values into the device memory.
  cudaMemcpy(d_params, x,
      t->nparams * sizeof(double), cudaMemcpyHostToDevice);

  // TODO be more clever about this.
  // Possibly copy not all of the data at once,
  // but also not only one row of data at a time.
  cudaMemcpy(d_predict_all, t->predict,
      t->nobs * t->npredict * sizeof(double), cudaMemcpyHostToDevice);

  // Allocate host memory.
  double *cat_exps = (double *) malloc(t->ncats * sizeof(double));

  // For each observation, do things per category.
  // This structure will be more easily generalized for streaming data.
  for (i_obs=0; i_obs<t->nobs; ++i_obs)
  {

    if ((i_obs+1) % 100000 == 0)
    {
      printf("row %d of %d\n", i_obs+1, t->nobs);
      fflush(stdout);
    }

    // Define the predictors and response for this observation.
    double *predict = t->predict + i_obs * t->npredict;
    double *d_predict = d_predict_all + i_obs * t->npredict;
    int respond = t->respond[i_obs];

    // TODO this is very inefficient to do inside this loop.
    // Copy the prediction array into the device memory.
    //cudaMemcpy(d_predict, predict,
        //t->npredict * sizeof(double), cudaMemcpyHostToDevice);

    // TODO remove this is the non-cuda code chunk.
    // Compute an exp of a dot product for each non-last category.
    /*
    for (i_cat=0; i_cat<t->ncats; ++i_cat)
    {
      compute_cat_exps(t, i_cat, cat_exps, x, predict);
    }
    */

    // Launch the CUDA kernel.
    compute_nonlast_cat_exps<<<1, t->ncats>>>(
        t->ncats,
        t->npredict,
        d_cat_exps,
        d_params,
        d_predict
        );

    // Copy the per-category exps for this data row back into the host memory.
    cudaMemcpy(cat_exps, d_cat_exps,
        t->ncats * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute the exp for the last category.
    // The parameters associated with this reference category are zero.
    cat_exps[t->ncats-1] = 1;

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
  fflush(stdout);

  free(cat_exps);

  cudaFree(d_cat_exps);
  cudaFree(d_params);
  cudaFree(d_predict_all);

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
    fflush(stdout);

    return 0;
}


int main(int argc, char *argv[])
{
  // Initialize some arguments to be read from the command line.
  int n = -1; // number of observations
  int d = -1; // number of categories including the reference category
  int k = -1; // number of predictors including the intercept
  int m = -1; // max number of memories (default is six)
  int max_iterations = -1; // max number of iterations (default is infinite)
  char *p_opt = 0; // predictor filename
  char *r_opt = 0; // response filename
  int verbose = 0; // verbosity

  // Read the command line arguments.
  int c;
  while ( (c = getopt(argc, argv, "vn:d:k:m:i:p:r:")) != -1)
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
      case 'm':
        m = atoi(optarg);
        break;
      case 'i':
        max_iterations = atoi(optarg);
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
    fflush(stdout);
  }
  int i, ret = 0;
  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(t.nparams);

  if (x == NULL) {
    printf("failed to allocate a memory block for variables\n");
    return -2;
  }

  // Initialize parameter values to zero.
  if (verbose)
  {
    printf("initialize parameter values to zero...\n");
    fflush(stdout);
  }
  for (i=0; i<t.nparams; ++i)
  {
    x[i] = 0;
  }

  /* Initialize the parameters for the L-BFGS optimization. */
  if (verbose)
  {
    printf("initialize parameters for L-BFGS optimization...\n");
    fflush(stdout);
  }
  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  if (m > 0)
  {
    param.m = m;
  }
  if (max_iterations > 0)
  {
    param.max_iterations = max_iterations;
  }
  /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

  /*
    Start the L-BFGS optimization; this will invoke the callback functions
    evaluate() and progress() when necessary.
   */
  if (verbose)
  {
    printf("starting L-BFGS loop\n");
    fflush(stdout);
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

