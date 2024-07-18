# GeneticDecisionTree
Decision Tree built using a genetic algorithm

## Motivation
It's often useful in machine learning to use interpretable models for prediction problems, either as the actual model, or as proxy models to approximate the behaviour of the actual (blackbox) models (providing a form of post-hoc explanation). Decision trees, at least when constrained to reasonable sizes, are quite comprehensible and are excellent interpretable models when they are sufficiently accurate. However, decision trees are not always highly accurate and can often be fairly weak, particularly relative to stronger models for tabular data such as CatBoost, XGBoost, and LGBM (which are themselves boosted ensembles of decision trees). As well, where decision trees are suffiently accurate, this accuracy is often acheived by allowing the trees to grow to a large size, thereby eliminating any interpretability.

The greedy approach used by Decision Trees is often quite sub-optimal, though does allow trees to be contructed very quickly. Historically, this was more pressing given lower-powered computer systems, and in a modern context can also be very useful, as it allows constructing large numbers of trees in models based on ensembles of decision trees. However, to create a single decision tree that is both accurate and interpretable (of a manageable size), using a greedy algorithm is very limiting. 

With standard decision trees, the selection of the feature and threshold at each node is a one-time decision: decision trees are limited to the choices made for split points once these splits are selected.  While the trees can (at lower levels) compensate for poor modeling choices higher in the tree, this will usually reduce interpretability and may not fully mitigate the effects of the choices of split points above. 

## Genetic Algorithms
Genetic algorithms typically proceed by starting with a number of candidate solutions to a problem, then iterating many times, with each iteration selecting the strongest candidates, removing the others, and creating a new set of candidate solutions. This may be done either by mutating (randomly modifying) an existing model or by combining two or more into a new model, simulating reproduction as seen in real-world evolutionary processes. In this way, over time, a set of progressively stronger candidates tends to emerge.

During this process, it's also possible to regularly generate completely new random models. Although these will not have had the benefit of mutations or combining, they may nevertheless, by chance, be as strong as some more evolved-solutions, though this is increasingly less likely as the candidates that are developed through the genetic process become increasingly evolved. 

Applied to the construction of decision trees, genetic algorithms create a set of candidate decision trees, select the best of these, mutate and combine these (with some new instances possibly doing both: deriving new offspring from multiple existing models and mutating these offspring at the same time). These steps may be repeated any number of times. 

Each time a new tree is generated from one or more existing trees, a similar tree is created, but slightly different: changing either the feature and threshold, or simply the threshold, used in one or more (but usually a small number of) internal nodes, generally leaving most internal nodes the same (the predictions in the leaf nodes must also be re-calculated whenever internal nodes are modified). 

This process can be slow, requiring many iterations before substantial improvements in accuracy are seen, but in this case, our interest is in interpretability and so we can assume all decision trees are reasonably small, likely with a maximum depth of 2 to 5. This allows progress to be made substantially faster than where we attempt to evolve large decision trees. 

There have been over time a number of proposals for genetic algorithms for decision trees. This solution has the benefit of providing python code on github, but is far from the first and many other solutions may work better for your projects. In particular, this solution, though reasonably efficient, has had only moderate performance optimizing (it does allow executing slower operations in parallel, for example) and is far slower than standard decision trees, particularly when executing over many iterations. However, testing has found using just 3 to 5 iterations is usually sufficient to realize substantial improvements for classification as compared to scikit-learn decision trees. Regression is a more difficult problem in this context, as covered below. But, for classification, this tool can be a very useful tool to try, often allowing accurate trees of fairly small sizes. 

## Other Approaches to Creating Stronger Decision Trees
Other work seeking to make Decision Trees more accurate and interpretable (accurate at a constained size) include [Optimal Sparce Decision Trees](https://arxiv.org/abs/1904.12847), oblique decision trees, oblivious decision trees, [AdditiveDecisionTrees](https://github.com/Brett-Kennedy/AdditiveDecisionTree), and various rule-based systems, such as in [imodels](https://github.com/csinva/imodels) and [PRISM-Rules](https://github.com/Brett-Kennedy/PRISM-Rules). While rules are not equivalent to decision trees, they may often be used in a similar way and offer similar levels of interpretability.

Some tools such as [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), [FormulaFeatures](https://github.com/Brett-Kennedy/FormulaFeatures), and [RotationFeatures](https://github.com/Brett-Kennedy/RotationFeatures) may be combined with GeneticDecisionTrees to create models that are more accurate still. 

## Implementation Details
DecisionTrees can be fairly sensitive to the data used for training. This often leads to overfitting, but with the GeneticDecisionTree, we take advantage of this to generate random candidate models (along with varying the random seeds used). Internally, GeneticDecisionTree generates a set of scikit-learn decision trees, which are then converted into a structure specific to GeneticDecisionTrees (which makes the subsequent mutation and combination operations simpler). To induce these scikit-learn decision trees, we fit them using different bootstrap samples of the original training data. 

We also vary the size of the samples, allowing for further diversity. The sample sizes are based on a logarithmic distribution, so we are effectively selecting a random order of magnitude. This is limited to a minimum of 128 rows and a maximum of two times the full training set size. Smaller sizes are more common than larger, but occasionaly larger sizes are used as well. 

The algorithm starts by creating a small set of decision trees generated in this way. It then iterates a specified number of times (five by default). Each iteration:
- Randomly mutates the top-scored models created so far (those best fit to the training data).
- Combines pairs of the top-scored models created so far. This is done in an exhaustive manner over all pairs of the top performing trees. 
- Generates additional random trees using scikit-learn and random bootstrap samples (less of these are generated each iteration, as it becomes more difficult to compete with the models that have experienced mutating and/or combining).
- Selects the top-performing trees before looping back for the next iteration

Each iteration, a significant number of trees are generated. Each is then evaluated on the training data. Standard decision trees are constructed in a purely greedy manner (though it is possible to constrain them and to prune them), considering only the information gain for each possible split at each internal node. With Genetic Decision Trees, on the other hand, the construction of each tree is partially or entirely random (the construction done by scikit-learn is non-random, but is based on random samples; the mutations are random), but the important decisions made during fitting (selecting the best models generated so far) relate to the fit of the tree as a whole to the available training data. This tends to generate a final result that fits the training better than a greedy approach allows.  

Despite the utiltiy of the genetic process, an interesting finding is that: even while not performing mutations or combinations each iteration (these operations are configurable and may be set to False to allow faster execution times), GeneticDecisionTrees tend to be more accurate than standard Decision Trees limited to the same (small) size. This is, though, as expected: simply by trying many sets of possible choices for the internal nodes in a decision tree, some will perform better than a single tree constructed in the normal greedy fashion.

Where mutations and combinations are enabled, though, generally after one or two iterations, the majority of the top-scored candidate Decision Trees (the trees that fit the training data the best) will be based on mutating and/or combining other strong models. That is, enabling mutating and combining does tend to generate stronger models.

In the end, the top performing tree is selected and is used for prediction. 

### Execution Time
This improvement in accuracy over standard decision trees does come at the cost of time, but for most datasets, fitting is still only about 1 to 5 minutes, depending on the size of the data and the parameters specified. This is quite slow compared to training standard decision trees, which is often under a second. Nevertheless, a few minutes can often be warranted to generate an interpretable model, particularly when creating an accurate, interpretable model can often be quite challenging. 

Limiting the number of iterations to only 1 or 2 can reduce the training time and can often still achieve strong results. As would likely be expected, there are diminishing returns over time using additional iterations, and some increase in the chance of overfitting. 

It may be reasonable in some cases to disable mutations or combinations and instead generate only a series of random trees based on random bootstrap samples. This approach simply produces a large number of small decision trees and selects the best-performing of these. Where sufficient accuracy can be achieved in this way, this may be all that's necessary. It is also possible to start with this as a baseline, and then test if additional improvements can be found enabling mutations and/or combinations. Where these are used, the model should be set to execute at least a few iterations.  

### Generating Random Decision Trees
Where the model is run without mutations or combinations, we have only a series of random trees, and consequently do not execute a genetic algorithm. It may, nevertheless, be reasonable to do this, particularly where execution time is important. Test results below show this often works as well, or nearly as well, as allowing mutations and combinations, and often better. 

Using this approach, we effectively model the data in a way similar to a RandomForest, which is also based on a set of decision trees, each trained on a random bootstrap sample. However, RandomForests will use all Decision Trees created and combine their predictions, while GeneticDecisionTree retains only the single, strongest of these Decision Trees. 

### Mutating
The mutating process currently supported by GeneticDecisionTree allows only modifying the thresholds used by internal nodes, keeping the feature used the same. Each mutation will select one internal node randomly and set the threshold to a new random value. This is surprisingly effective and can often substantially change the training data used in the two child nodes below it (and consequently the two sub-trees below the selected node). The trees start with the thresholds assigned by scikit-learn, selected based on information gain only (not considering the tree as a whole). Even holding the remainder of the tree constant, modifying these thresholds can effectively induce quite different trees. 

Future versions may allow rotating nodes within the tree, but testing to date has found this not as effective as simply modifying the thresholds for a single internal node. However, more research will be done on other mutations that may prove effective and efficient.

### Combining
The other form of modification currently supported is combining two parent decision trees. To do this, we take the top 20 trees found during the previous iteration and attempt to combine each pair of these. A combination is possible if the two trees use the same feature in their root nodes. For example, if Tree 1 has a split in its root node on Feature D > 10.4 and Tree 2 has a split in its root node on Feature D > 10.8, then we can combine the two trees. We actually create two new trees. In both cases, the split in the root node is taken as the average of that in the thresholds of the two parents, so in this example, both new trees will have Feature D > 10.6 in their root nodes.

The first new tree will have Tree 1's left sub-tree (the left sub-tree under Tree 1's root node) and Tree 2's right sub tree. The other new tree will have Tree 2's left sub-tree and Tree 1's right sub-tree.

Future versions will allow combining using nodes other than the root, though the effects are smaller in these cases. 

### Overfitting
Decision Trees commonly overfit and GeneticDecisionTrees may as well. Like most models, GeneticDecisionTree attempts to fit to the training data as well as is possible, which may cause it to generalize poorly compared to other decision trees of the same size. However, overfitting is limited as the tree sizes are generally quite small, and the trees cannot grow beyond the specified maximum depth. Each candidate decision tree produced will have equal complexity (or nearly -- some paths may not extend to the full maximum depth allowed), so are roughly equally likely to overfit. As with any model, it's recommended to tune GeneticDecisionTrees to find the model that appears to work best with your data.

## Regression
GeneticDecisionTrees support both classification and regression, but are more appropriate for classification. Regression functions are very difficult to model with shallow decision trees, as it's necessary to predict a numeric value and each leaf node predicts only a single value. For example, a tree with 8 leaf nodes can predict only 8 unique values. This is often quite sufficient for classification problems (assuming the number of classes is under 8) but can produce only very approximate predictions with regression. With regression problems, even with simple functions, generally very deep trees are necessary to produce accurate results. Using a small tree with regression is viable only where the data has only a small number of distinct values in the target column, or where the values are in a small number of clusters, with the range of each being fairly small. 

GeneticDecisionTrees can work setting the maximum depth to a very high level, allowing accurate models, often substantially higher than standard decision trees, but the trees will not be interpretable, and the accuracy, while often strong, will likely not be competitive with strong models such as XGBoost, LGBM, or CatBoost.

## Examples
The following example is taken from the Simple_Examples notebook. This loads a dataset, does a train-test split, fits a GeneticDecisionTree, creates predictions, and outputs the accuracy, here using the F1 macro score. 

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_wine

from genetic_decision_tree import GeneticDecisionTree

data = load_wine()
df = pd.DataFrame(data.data)
df.columns = data.feature_names
y_true = data.target

X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.3, random_state=42)

gdt = GeneticDecisionTree(max_depth=2, max_iterations=5, allow_mutate=True, allow_combine=True, verbose=True)
gdt.fit(X_train, y_train)
y_pred = gdt.predict(X_test)
print("Genetic DT:", f1_score(y_test, y_pred, average='macro'))
```
GeneticDecisionTree is a single class used for both classification and regression. It infers from the target data the data type and handles the distinctions between regression and classification internally. 

## Example Notebooks
**[Simple Example](https://github.com/Brett-Kennedy/GeneticDecisionTree/blob/main/Examples/Simple_Examples.ipynb)**

This provides a small number of examples with real (toy) and synthetic data, both for classification and regression. 

The first example uses the Wine dataset, available with scikit-learn. Using a depth of 2, GeneticDecisionTree achieves an F1 macro score on a hold-out test set of 0.97 as compared to 0.88 for a standard decision tree.

GeneticDecisionTrees provide an export_tree() method similar to scikit-learn Decision Trees. In this case, it produces a very easily-understood tree:
```
IF flavanoids < 1.4000
| IF color_intensity < 3.7250
| | 1
| ELSE color_intensity > 3.7250
| | 2
ELSE flavanoids > 1.4000
| IF proline < 724.5000
| | 1
| ELSE proline > 724.5000
| | 0
```
For the most part, the examples use max_depth=4 as this tends to achieve high performance while being reasonably interpretable. 

**[TestClassification](https://github.com/Brett-Kennedy/GeneticDecisionTree/blob/main/Examples/Test_Classification.ipynb)**

This provides an extensive test of GeneticDecisionTrees. It tests with a large number of test sets from OpenML and for each creates a standard Decision Tree and 4 GeneticDecisionTrees: each combination of allowing mutations and allowing combinations (supporting neither, mutations only, combinations only, and both).

In almost all cases, at least one, and often all four, variations of the GeneticDecisionTree strongly out perform the standard decsision tree, again using F1 macro scores. A subset of this is shown here:

![img](https://github.com/Brett-Kennedy/GeneticDecisionTree/blob/main/Images/img1.png)

Given the large number of cases tested, running this notebook is quite slow. It is also not a defintive evaluation. It uses only a limited set of test files, uses only default parameters other than max_depth, and tests only the F1 macro scores. It does, however, demonstrate the GeneticDecisionTrees can be effective and interpretable models in many cases. 

**[TestRegression](https://github.com/Brett-Kennedy/GeneticDecisionTree/blob/main/Examples/Test_Regression.ipynb)**

This provides another example using Genetic Decision Trees for regression, but as indicated, the gains will tend to be minimal if there are any. 

## Tuning
GeneticDecisionTrees have few parameters, simplifying tuning. In fitting, we specify only the maximum depth of the final tree, the number of iterations, and if we allow mutating or combining. In general, more iterations is preferred with respect to accuracy, though the additional work beyond a few iterations may not produce substantial improvements. A random_state parameter is also provided to allow tuning this as well, which may result randomly in better candidate models being produced or better mutations being performed. 

In one of the synthetic datasets included in the sample notebook, the tree generated looks like:

```
IF c < 0.9016
| IF b < 0.9098
| | IF a < 0.6904
| | | IF d < 0.7046
| | | | W
| | | ELSE d > 0.7046
| | | | W
```
In this case, the true function is known, and the actual thresholds are 0.9, 0.7, and so on, indicating the threshods discovered are very close, but random changes followed be evaluation may push them to more accurate values. 

## Installation
The tool is contained in a single .py file, [genetic_decision_tree.py](https://github.com/Brett-Kennedy/GeneticDecisionTree/blob/main/genetic_decision_tree.py), which may simply be downloaded and imported into any project. It uses only standard python libraries. 


## API

### GeneticDecisionTree

```python
    gdt = GeneticDecisionTree(
        max_depth=4,
        max_iterations=10,
        allow_mutate=True,
        allow_combine=True,
        n_jobs=1,
        verbose=False)
```

**Parameters**

**max_depth**: int

The maximum depth the tree may grow to. The smaller, the more interpretable, but typically less accurate.

**max_iterations**: int

The number of iterations the fit algorithm will execute. Each iteration, a number of trees are created (either randomly or based on the previously-created trees). At the end of each iteration, the top trees are kept and taken as the starting point for the next iteration.

**allow_mutate**: bool

May be set False to reduce time and potentially overfitting. If True, variations of existing trees will be created based on modifying the thresholds of random nodes to random values.

**allow_combine**: bool

May be set False to reduce time and potentially overfitting. If True, combinations of existing nodes will be created taking the left sub-tree of one node and right sub-tree on another node.

**n_jobs**: int

Controls the number of processes that may be created. 

**random_state**: int

Used to make any random processes repeatable.

**verbose**: bool

If set True, some output will be displayed during the fitting processes.

##

### fit

```python
gdt.fit(x, y)
```

Fits the model to the training data provided. Internally, this generates a set of decision trees and selects the best-peforming with respect to the training data.

##

### predict()

```python
y_pred = gdt.predict(x)
```
Returns a prediction for each element in x.


##

### export_tree()

```python
gdt.export_tree(x)
```
Draws a representation of the final decision tree discovered during fitting and used during prediction. 

### Verbose Output
The fit process allows verbose output. If this is enabled, each iteration we see:
- the iteration count
- The scores of the top trees discovered so far. This is the training scores and may be over-optimistic relative to the test scores. It is shown to indicate the progress of the fitting process.
- The number of the top trees discovered so far that are either:
  - mutated variations of other trees or are descended (through combinations) from one or more trees that included mutations
  - combinations of other trees or are descended (through mutation) from one or more other trees that were combinations



