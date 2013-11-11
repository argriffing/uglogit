
```
iris data:

150 observations
5 predictors including the intercept
3 categories including the reference category

$ od --format=f8 --width=40 --output-duplicates iris.predict
$ od --format=d4 --width=4 --output-duplicates iris.respond
```

```
how to build:

$ gcc -o ulogit uglogit.c <path-to>/lbfgs.o -I<path-to>/liblbfgs/include -lm
```
