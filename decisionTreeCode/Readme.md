Decision Tree Module

This module describes code for a decision tree class, as well
as those of randomForest and a variable ranker using random forest
and decision trees.

The input are data points, with a distinguished index to try and predict.
Also desired are the set of positive classes for splitting on attributes
in the random forest importance function.

The variable ranker will grow a forest and then predict a bootstrap sample
along with a permuted sample by variable x_i in order to measure the difference
in classification accuracy and thus estimate the importance of variable x_i.


