# GeneticDecisionTree
Decision Tree built using a genetic algorithm

## Motiviation
It's often useful in machine learning to use interpretable models for prediction problems, either as the actual model, or to use as proxy models to approximate the behaviour of the actual (blackbox) models. Decision trees, at least when constrained to reasonable sizes are quite comprehensible, and are excellent interpretable models when they are sufficiently accurate. However, decision trees are often inaccurate, at least relative to stronger models for tabular data such as CatBoost, XGBoost, and LGBM (which are themselves boosted ensembled of decision trees). As well, where decision trees are suffiently accurate, this accuracy is often acheived by allowing the tree to grow to a large size, thereby eliminating any interpretability.

The greedy approach used by Decision Trees is often quite sub-optimal, though does allow trees to be contructed very quickly. Historically, this was more pressing given lower-powered computer systems, and in a modern context, can be very useful, as it allows constructing large numbers of decision trees in models based on ensembles of decision trees. However, to create a single decision tree that is both accurate and interpretable (of a manageable size), using a greedy algorithm is very limiting. 

With standard decision trees, the selection of the feature and the threshold at each node is a one-shot deal. While the trees can compensate for poor modeling choices lower in the tree, this will usually reduce interpretability and may not fully mitigate the effects of the choices of split points above. Decision trees are stuck with the choices for split points once these splits are selected. 

## Genetic Algorithsm
Genetic algorihms typically proceed by starting with a number of candidate solutions to a problem, then iterating many times, with each iteration selecting the strongest candidates, removing the others, and creating a new set of candidate solutions. This may be done either by mutating (randomly modifying) and existing model or by combining two or more into a new model, simulating sexual reproduction as seen in real-world evolutionary processes. In this way, over time, as set of progressively stronger tends to emerge.

It's also possible to regularly generate completely new random models. Although these have not had the benefit of mutations or combining, they many nevertheless, by chance, be as strong as some more evolved-solutions, though this is progressively less likely. 

Applied to the construction of decision trees, genetic algorithms create a set of candidate decision trees, select the best of these, mutate and combine these (possibly both deriving new offspring from multiple existing models and mutating these). This steps may be repeated any number of times, each time changing either the feature (and threshold), or simply the threshold, used in one or more nodes. 

This process can be slow, requiring many iterations before substantial improvements in accuracy are seen, but in this case, our interest is in interpretability, and so we can assume all decision trees are reasonably small, likely with a maximum depth of 2 to 5. 

There have been over time a number of proposals for genetic algorithms for decision trees. This solution provides python code on github, but is far from the first and many other solutions may work better for your projects. In particular, this solution, though reasonably efficient, has had only moderate performance optimizing and is far slower than standard decision trees, particularly when executing over many iterations. However, testing has found using just 3 to 5 iterations is usually quite sufficient to realize substantial improvements for classification as compared to scikit-learn decision trees. Regression is a more difficult problem in this context, as covered below. But, for classification, this tool can be a very useful tool to try, often allowing accurate trees of fairly small sizes. 

## Other Approaches to Creating Stronger Decision Trees
Other work seeking to make Decision Trees more accurate and interpretable (accurate at a constained size) includ Optimal Sparce Decision Trees, oblique decision trees, AdditiveDecisionTrees, FormulaFeatures, and various rule-based systems, such as in imodels. While rules are not equivalent to decision trees, they may often be used in a similar way and offer similar levels of interpretability.

Some tools such as ArithmeticFeatures, FormulaFeatures, RotationFeatures may be combined with GeneticDecisionTrees to create models that are more accurate still. 

DTs can be fairly sensitive to the training data, so by using different bootstrap samples, can induce different trees. Also set the random_state differently each time. Doing this can generate a large number of trees and then test the trees as a whole. That is, decision trees are constructed in a greedy manner (though it is possible to constrain them and to prune them), which means each decision considers only the current split and this is based only on the data in this subspace. With Genetic Decision Trees, on the otherhand, the construction is largely random (other than the construction done by scikit-learn decision trees (which are used internally for some tree generation), but the decisions make during fitting relate to the fit of the tree as a whole to the available training data.

there are many papers discussing creating decision trees using genetic algorithms. This is just one example, but does have a python implementation on github.


## Implementation Details
interestingly, simply creating many dts based on boostrap samples and taking the best often works the best or nearly, and is quite fast.

This also does mutations based on adjusting the thresholds. Can be set at one splitpoint during construction, but given the sub-trees built underneath them, a different can often work better, though typically only slightly. This works by taking the top 10 trees created so far, and creating 50 variations on each: picking 5 random nodes, and for each of these, 10 new thresholds.

And does combinations. Where two trees among the top 20 have the same feature in the root node, will create 2 combinations: one with the left sub-tree from the first parent tree and the right sub-tree from the second parent tree, and one that's the reverse. 

Genetic Decision Trees are slower. But manageable. Can disable creating mutations or combinations if want faster. Then will just create random trees based on bootstrap samples.

### Generating Random Decision Trees
Uses log scale to get size of sample. can be up to 2* the actual size, but usually much smaller. At least 128.

if disable mutations and combinations, just have random trees, so not really a genetic algorithm in this case, but can nevertheless generate strong trees, and is much faster. Is worth trying both. 

If we just use random, it's a bit like a RF. Many trees based on bootstrap sample. Though, uses the best tree instead of an ensemble of many. 


### Mutating
can also rotate nodes, but doesn't tend to work well. future work may improve this. 

### Combining

### Overfitting
Decision Trees commonly overfit, and GeneticDecisionTrees may as well. 

like most models, tries to fit the training data as well as it can, so can overfit. but the idea is to keep small, so tends not to drastically overfit. 

## Regression

this supports regression, but it's difficult to have high accuracy with a shallow decision tree, even with a simple function. you need to zero in on the exact values. Possible only if the target column has relatively few unique values, or many, but in a small number of ranges. 

can work unconstrained. but then not interpretable. may be more accurate, but likely better to use an ensemble of trees, such as RandomForest, XGBoost, CatBoost, ExtraTrees, or others.


## Examples

Just a single class -- it infers from the target data the data type and handles the distinctions between regression and classification internally. 

## Example Notebooks

we focus on max_depth=4. For interpretable, probably want 3, 4, or 5.

give a full example of the synth so can see the full tree. 


Running the test notebook is very slow. Tests many files.

Not a definitive evaluation, uses a small set of datasets, default parameters other than max_depth, and uses only f2 macro. 

## Tuning
Tuning
Is easy to tune since few parameters.
can try running with no mutating or combining. That's not really genetic, but can be fine, and is much faster. If good enough, you're done. If not, can run longer, or can try mutating & combining. Usually get a bit better results.
usually more iterations is better, but is more inclined to overfit. Does not get more complex though, just tries more combinations. 

With the synthetic data, the tree found looks like:
IF c < 0.9016
| IF b < 0.9098
| | IF a < 0.6904
| | | IF d < 0.7046
| | | | W
| | | ELSE d > 0.7046
| | | | W

We can see the thresholds are close to the true thresholds, but not quite. Adjust may be helpful.

## Installation


## API


### Verbose Output
describe this



