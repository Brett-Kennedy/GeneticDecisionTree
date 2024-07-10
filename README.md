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
