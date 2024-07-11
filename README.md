# GeneticDecisionTree
Decision Tree built using a genetic algorithm


the greedy approach used by DTs is far from optimal. It is fast, which was necessary historically with slower computers, even with usually smaller data. And is useful in bagged & boosted ensembles (eg RandomForest, ExtraTrees, CatBoost, XGBoost, LGBM, ect.)

but is not optimal, and often other trees of the same size can be more accurate.

we focus on max_depth=4. For interpretable, probably want 3, 4, or 5.

interestingly, simply creating many dts based on boostrap samples and taking the best often works the best or nearly, and is quite fast.

This also does mutations based on adjusting the thresholds. Can be set at one splitpoint during construction, but given the sub-trees built underneath them, a different can often work better, though typically only slightly. This works by taking the top 10 trees created so far, and creating 50 variations on each: picking 5 random nodes, and for each of these, 10 new thresholds.

And does combinations. Where two trees among the top 20 have the same feature in the root node, will create 2 combinations: one with the left sub-tree from the first parent tree and the right sub-tree from the second parent tree, and one that's the reverse. 

Running the test notebook is very slow. Tests many files.

Genetic Decision Trees are slower. But manageable. Can disable creating mutations or combinations if want faster. Then will just create random trees based on bootstrap samples.

Is other work trying to make DTs more reliable, including Optimal Sparce Decision Trees, oblique decision trees, oblivious trees, AdditiveDecisionTrees, FormulaFeatures. 

DTs are naturally quite interpretable if kept to small size but are still accurate, so it's natural much of the work in interpretable AI, including my own, has worked with trying to make decision trees more accurate and interpretable. 

DTs can be fairly sensitive to the training data, so by using different bootstrap samples, can induce different trees. Also set the random_state differently each time. Doing this can generate a large number of trees and then test the trees as a whole. That is, decision trees are constructed in a greedy manner (though it is possible to constrain them and to prune them), which means each decision considers only the current split and this is based only on the data in this subspace. With Genetic Decision Trees, on the otherhand, the construction is largely random (other than the construction done by scikit-learn decision trees (which are used internally for some tree generation), but the decisions make during fitting relate to the fit of the tree as a whole to the available training data.

if disable mutations and combinations, just have random trees, so not really a genetic algorithm in this case, but can nevertheless generate strong trees, and is much faster. Is worth trying both. 

Mutating
can also rotate nodes, but doesn't tend to work well. future work may improve this. 

Random
Uses log scale to get size of sample. can be up to 2* the actual size, but usually much smaller. At least 128.

Overfitting
like most models, tries to fit the training data as well as it can, so can overfit. but the idea is to keep small, so tends not to drastically overfit. 

Evaluate
I can use DatasetsEvaluator. I'll try to make it internally parallel though, then just loop through a bunch of files, which is maybe easier.

Tuning
Is easy to tune since few parameters.
can try running with no mutating or combining. That's not really genetic, but can be fine, and is much faster. If good enough, you're done. If not, can run longer, or can try mutating & combining. Usually get a bit better results.

With the synthetic data, the tree found looks like:
IF c < 0.9016
| IF b < 0.9098
| | IF a < 0.6904
| | | IF d < 0.7046
| | | | W
| | | ELSE d > 0.7046
| | | | W

We can see the thresholds are close to the true thresholds, but not quite. Adjust may be helpful.

If we just use random, it's a bit like a RF. Many trees based on bootstrap sample. Though, uses the best tree instead of an ensemble of many. 

Verbose Output
describe this
