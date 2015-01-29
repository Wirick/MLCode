This repository contains various code that is applicable to machine learning.

I'm going to be doing some ML work now.
So I'm going to be doing a lot
of math, and a lot of ml fun.

The Probability Model can create uniform, single
and multivariable normal, categorical, and mixture model samples from
distributions with a specified mean and (co)variance. the plotsample.py is
a script that I wrote with plotting functions so that I could test my models
by eye. I use the Box Muller transform and the Cholesky decomposition for normal samples.

regression.py is the skeleton for the first assignment for UC Berkeley's 2011 machine
learning course. It does k-local linear regression on a location, phase, station pair
to predict the time residual of a seismic event. other code is a precomputation procedure
to determine the k-local linear regression. It is too big for my computer to manage without
being painfully slow, so now I think I'll have to use some other trick up my sleeve to try and
work it out.

util.py contains helper functions. K-nearest neighbors, k-fold cross validation, and 
some functions created to make a toy test for my cross-validation function, which predicted
a plane nonlinear equation using nearest neighbors

plotregression.py is a visual test for my cross validation functions.
