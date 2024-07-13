# GeneticDecisionTree
Decision Tree built using a genetic algorithm

## Motiviation
It's often useful in machine learning to use interpretable models for prediction problems, either as the actual model, or to use as proxy models to approximate the behaviour of the actual (blackbox) models (providing a form of post-hoc explanation). Decision trees, at least when constrained to reasonable sizes, are quite comprehensible and are excellent interpretable models when they are sufficiently accurate. However, decision trees are often inaccurate, at least relative to stronger models for tabular data such as CatBoost, XGBoost, and LGBM (which are themselves boosted ensembled of decision trees). As well, where decision trees are suffiently accurate, this accuracy is often acheived by allowing the tree to grow to a large size, thereby eliminating any interpretability.

The greedy approach used by Decision Trees is often quite sub-optimal, though does allow trees to be contructed very quickly. Historically, this was more pressing given lower-powered computer systems, and in a modern context, can also be very useful, as it allows constructing large numbers of decision trees in models based on ensembles of decision trees. However, to create a single decision tree that is both accurate and interpretable (of a manageable size), using a greedy algorithm is very limiting. 

With standard decision trees, the selection of the feature and the threshold at each node is a one-shot deal. While the trees can (at lower levels) compensate for poor modeling choices higher in the tree, this will usually reduce interpretability and may not fully mitigate the effects of the choices of split points above. Decision trees are limited to the choices made for split points once these splits are selected. 

## Genetic Algorithms
Genetic algorithms typically proceed by starting with a number of candidate solutions to a problem, then iterating many times, with each iteration selecting the strongest candidates, removing the others, and creating a new set of candidate solutions. This may be done either by mutating (randomly modifying) an existing model or by combining two or more into a new model, simulating sexual reproduction as seen in real-world evolutionary processes. In this way, over time, a set of progressively stronger candidates tends to emerge.

During this process, it's also possible to regularly generate completely new random models. Although these will not have had the benefit of mutations or combining, they many nevertheless, by chance, be as strong as some more evolved-solutions, though this is increasingly less likely and more evolved solutions are developed through the genetic process. 

Applied to the construction of decision trees, genetic algorithms create a set of candidate decision trees, select the best of these, mutate and combine these (wiht some new instances possibly doing both: deriving new offspring from multiple existing models and mutating these offspring at the same time). These steps may be repeated any number of times. Each time a new tree is generated from one or more existing trees, a similar tree is created, but slightly different: changing either the feature and threshold, or simply the threshold, used in one or more nodes, generally leaving most internal nodes the same (the predictions in the leaf nodes must also be re-calculated whenever internal nodes are modified). 

This process can be slow, requiring many iterations before substantial improvements in accuracy are seen, but in this case, our interest is in interpretability and so we can assume all decision trees are reasonably small, likely with a maximum depth of 2 to 5. 

There have been over time a number of proposals for genetic algorithms for decision trees. This solution provides python code on github, but is far from the first and many other solutions may work better for your projects. In particular, this solution, though reasonably efficient, has had only moderate performance optimizing and is far slower than standard decision trees, particularly when executing over many iterations. However, testing has found using just 3 to 5 iterations is usually quite sufficient to realize substantial improvements for classification as compared to scikit-learn decision trees. Regression is a more difficult problem in this context, as covered below. But, for classification, this tool can be a very useful tool to try, often allowing accurate trees of fairly small sizes. 

## Other Approaches to Creating Stronger Decision Trees
Other work seeking to make Decision Trees more accurate and interpretable (accurate at a constained size) includ Optimal Sparce Decision Trees, oblique decision trees, AdditiveDecisionTrees, FormulaFeatures, and various rule-based systems, such as in imodels. While rules are not equivalent to decision trees, they may often be used in a similar way and offer similar levels of interpretability.

Some tools such as ArithmeticFeatures, FormulaFeatures, RotationFeatures may be combined with GeneticDecisionTrees to create models that are more accurate still. 

there are many papers discussing creating decision trees using genetic algorithms. This is just one example, but does have a python implementation on github.

## Implementation Details
DecisionTrees can be fairly sensitive to the data used for training. This often leads to overfitting, but with the GeneticDecisionTree, we take advantage of this to generate random candidate models (along with varying the random seeds used). Internally, GeneticDecisionTree generates a set of scikit-learn decision trees, which are then converted into a structure specific to GeneticDecisionTrees (which makes the mutation and combination operations later simpler). To induce these scikit-learn decision trees, we fit them using different bootstrap samples of the original training data. 

We also vary the size of the samples, allowing for more diversity. The sample sizes are based on a logarithmic distribution, so we are effectively selecting a random order of magnitude. This is limited to a minimum of 128 rows and a maximum of two times the full training set size. Smaller sizes are more common, but occasionaly larger sizes are used as well. 

The algorithm starts by creating a small set of decision trees generated in this way. It then iterates a specified number of times (five by default), each iteration:
- Randomly mutating the top-scored models created so far (those best fit to the training data).
- Randomly combining two of the top-scored models created so far
- Generating additional random trees using scikit-learn and random bootstrap samples (less of these are generated each iteration, as it becomes more difficult to compete with the models that have experience mutating and/or combining).

Each iteration, a significant number of trees are generated. Each is then evaluated on the training data. Standard decision trees are constructed in a purely greedy manner (though it is possible to constrain them and to prune them), considering only the information gain for each possibly split at each internal node. With Genetic Decision Trees, on the otherhand, the construction is largely random (other than the construction done by scikit-learn decision trees, which are used internally for some tree generation), but the decisions made during fitting relate to the fit of the tree as a whole to the available training data. This tends to generate a final result that fits the training better than a greedy approach allows. 

Having said this, an interesting finding is that, even while not performing mutations or combinations each iteration (these operations are configurable and may be set to False to allow faster execution times), GeneticDecisionTrees tend to be more accurate than standard Decision Trees limited to the same (small) size. This is, though, as expected: simply by trying many sets of possible choices for the internal nodes in a decision tree, some will perform better than a tree constructed in the normal greedy fashion.

Where mutations and combinations are enabled, generally after one or two iterations, most of the top-scored candidate Decision Trees (the trees that fit the training data the best) will be based on mutating and/or combining other strong models. That is, enabling mutating and combining does tend to generate stronger models.

## Execution Time
This improvement in accuracy over standard decision trees does come at the cost of time, but for most datasets, fitting is still only 1 to 5 minutes, depending on the parameters specified. This is compared to training standard decision trees, which is often under a second. Nevertheless, a few minutes can often be warranted to generate an interpretable model, particularly when creating an accurate, interpretable model can often be quite challenging. 

Limiting the number of iterations to only 1 or 2 can often still achieve strong results. As expected, there are diminishing returns over time, and some increase in the chance of overfitting. 

It may be reasonable in some cases to disable mutations or combination and instead generate only a series of random trees based on random bootstrap samples. Where sufficient accuracy can be achieved in this way, this may be all that's necessary. 

### Generating Random Decision Trees
Where the model is run without mutations or combinations, we have only a series of random trees, and consequently do not execute a genetic algorithm. It may, nevertheless, be reasonable to do this, particularly where execution time is important. Test results below show this often works as well or nearly as allowing mutations and combinations. 

Using this approach, we effectively model the data in a way similar to a RandomForest, which is also based on a set of decision trees, each trained on a random bootstrap sample. However, RandomForests will use all Decision Trees created and combine their predictions, while GeneticDecisionTree retains only the single, strongest of these DecisionTrees. 


### Mutating
can also rotate nodes, but doesn't tend to work well. future work may improve this. 

This also does mutations based on adjusting the thresholds. Can be set at one splitpoint during construction, but given the sub-trees built underneath them, a different can often work better, though typically only slightly. This works by taking the top 10 trees created so far, and creating 50 variations on each: picking 5 random nodes, and for each of these, 10 new thresholds.


### Combining
And does combinations. Where two trees among the top 20 have the same feature in the root node, will create 2 combinations: one with the left sub-tree from the first parent tree and the right sub-tree from the second parent tree, and one that's the reverse. 

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



