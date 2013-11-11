
```
iris data:

150 observations
5 predictors including the intercept
3 categories including the reference category

$ od --format=f8 --width=40 --output-duplicates iris.predict
$ od --format=d4 --width=4 --output-duplicates iris.respond

$ ./uglogit -n 150 -d 3 -k 5 -p iris.predict -r iris.respond
```

```
aa data:

34009810 observations
5 predictors including the intercept
400 categories including the reference category

$ ./uglogit -n 34009810 -d 400 -k 5 -p aa.predict -r aa.respond
```

```
how to build:

$ gcc -o uglogit uglogit.c <path-to>/lbfgs.o -I<path-to>/liblbfgs/include -lm
```
