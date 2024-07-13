import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, r2_score
import concurrent
from functools import partial
import datetime
from pandas.api.types import is_numeric_dtype


class InternalDecisionTree:
    """
    A simple decision tree used internally. The GeneticDecisionTree will create many instances of this during
    fitting and then take the best-performing of these, which will be used for prediction.
    """

    def __init__(self, source_desc, classes_, target_type):
        self.source_desc = source_desc
        self.target_type = target_type
        self.classes_ = classes_

        # Parallel arrays related to the nodes generated
        self.feature = None
        self.threshold = None
        self.children_left = None
        self.children_right = None
        self.node_prediction = None

    def copy_from_dt(self, dt):
        """
        Used to copy the contents of an scikit-learn decision tree
        """
        self.feature = dt.tree_.feature
        self.threshold = dt.tree_.threshold
        self.children_left = dt.tree_.children_left
        self.children_right = dt.tree_.children_right

        if self.target_type == 'classification':
            # While scikit-learn classification decision trees store a probability for each class in the value array,
            # InternalDecisionTree stores only the prediction (the most frequent class from the training data) for each
            # element of node_prediction.
            # The dt may have the classes in a different order from the GeneticDecisionTree, but here we take the
            # predictions based on the order in the dt.
            self.node_prediction = [dt.classes_[np.argmax(x)] for x in dt.tree_.value]
        else:
            self.node_prediction = [x[0][0] for x in dt.tree_.value]

    def copy_from_values(self, feature, threshold, children_left, children_right):
        self.feature = feature
        self.threshold = threshold
        self.children_left = children_left
        self.children_right = children_right
        self.node_prediction = None

    def count_training_records(self, x, y):
        counts = np.zeros((len(self.feature), len(self.classes_)))

        # Loop through every record. For each, determine which leaf node it ends in. For that leaf node increment
        # by one the count for the value of Y for this record.
        for i in x.index:
            y_index = self.classes_.index(y.loc[i])
            row = x.loc[i]
            cur_node_idx = 0
            while self.feature[cur_node_idx] >= 0:
                cur_feature_name = x.columns[self.feature[cur_node_idx]]
                cur_threshold = self.threshold[cur_node_idx]
                if row[cur_feature_name] >= cur_threshold:
                    cur_node_idx = self.children_right[cur_node_idx]
                else:
                    cur_node_idx = self.children_left[cur_node_idx]
            counts[cur_node_idx][y_index] += 1

        # Now that we have counts, determine the prediction for each leaf node
        self.node_prediction = [""]*len(self.feature)
        for node_idx in range(len(self.feature)):
            if self.feature[node_idx] == -2:
                self.node_prediction[node_idx] = self.classes_[np.argmax(counts[node_idx])]

    def find_means(self, x, y):
        sums_arr = [0.0]*len(self.feature)
        counts_arr = [0]*len(self.feature)

        # Loop through every record. For each, determine the node it ends in. For that node, update the sum and the
        # count of the y values for that node.
        for i in x.index:
            row = x.loc[i]
            cur_node_idx = 0
            while self.feature[cur_node_idx] >= 0:
                cur_feature_name = x.columns[self.feature[cur_node_idx]]
                cur_threshold = self.threshold[cur_node_idx]
                if row[cur_feature_name] >= cur_threshold:
                    cur_node_idx = self.children_right[cur_node_idx]
                else:
                    cur_node_idx = self.children_left[cur_node_idx]
            sums_arr[cur_node_idx] += y.loc[i]
            counts_arr[cur_node_idx] += 1

        self.node_prediction = [-1]*len(self.feature)
        for node_idx in range(len(self.feature)):
            if counts_arr[node_idx] > 0:
                self.node_prediction[node_idx] = sums_arr[node_idx] / counts_arr[node_idx]

    def predict(self, x):
        ret = []
        for i in x.index:
            row = x.loc[i]
            cur_node_idx = 0
            while self.feature[cur_node_idx] >= 0:
                cur_feature_name = self.feature[cur_node_idx]
                cur_threshold = self.threshold[cur_node_idx]
                if row[cur_feature_name] >= cur_threshold:
                    cur_node_idx = self.children_right[cur_node_idx]
                else:
                    cur_node_idx = self.children_left[cur_node_idx]

            ret.append(self.node_prediction[cur_node_idx])
        return ret


class GeneticDecisionTree:
    def __init__(self,
                 max_depth=4,
                 max_iterations=10,
                 allow_mutate=True,
                 allow_combine=True,
                 n_jobs=1,
                 random_state=0,
                 verbose=False):
        """
        Create an instance of a GeneticDecisionTree. This may be used for regression or classification.

        :param max_depth: int
            The maximum depth the tree may grow to. The smaller, the more interpretable, but typically less accurate.

        :param max_iterations: int
            The number of iterations the fit algorithm will execute. Each iteration, a number of trees are created
            (either randomly or based on the previously-created trees). At the end of each iteration, the top trees
            are kept and taken as the starting point for the next iteration.

        :param allow_mutate: bool
            May be set False to reduce time and potentially overfitting. If True, variations of existing trees will
            be created based on modifying the thresholds of random nodes to random values.

        :param allow_combine: bool
            May be set False to reduce time and potentially overfitting. If True, combinations of existing nodes will
            be created taking the left sub-tree of one node and right sub-tree on another node.

        :param n_jobs: int
            Controls the number of processes that may be created. Currently supports only 1 and -1.

        :param random_state: int
            Used to make any random processes repeatable.

        :param verbose: bool
            If set True, some output will be displayed during the fitting processes.
        """

        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.allow_mutate = allow_mutate
        self.allow_combine = allow_combine
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.column_names = None
        self.classes_ = None
        self.internal_dt = None
        self.target_type = None

        # Ensure n_jobs is in the range that may be used by the concurrent library.
        if n_jobs == -1:
            self.n_jobs = 61
        if (self.n_jobs < 0) or (self.n_jobs > 61):
            print("n_jobs may be only -1 or a positive value <= 61. Setting to 1.")
            self.n_jobs = 1

    def fit(self, x, y):
        """
        Similar to standard models, fits the model to the data as well as possible given the maximum size specified.
        The fitting process for GeneticDecisionTrees includes creating many candidate trees and optionally mutating
        and combining these to identify the tree best fit to the training data.

        :param x: 2d matrix of values. Should not include the target.
        :param y: 1d array of target values.
        :return: None
        """

        def get_dt_scores():
            if self.n_jobs != 1:
                scores_list = []
                func = None
                if self.target_type == 'classification':
                    func = partial(get_scores_classification, x, y)
                else:
                    func = partial(get_scores_regression, x, y)
                idt_chunks = []
                chunk_size = 100
                s = datetime.datetime.now()
                for i in range(0, len(idt_list), chunk_size):
                    idt_chunks.append(idt_list[i: i+chunk_size])
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = executor.map(func, idt_chunks)
                for result in results:
                    scores_list.extend(result)
                return scores_list
            else:
                s = datetime.datetime.now()
                if self.target_type == 'classification':
                    scores_list = get_scores_classification(x, y, idt_list)
                else:
                    scores_list = get_scores_regression(x, y, idt_list)
                return scores_list

        def combine_trees(parent_a, parent_b):
            idt = InternalDecisionTree(
                parent_idt_1.source_desc + " & " + parent_idt_2.source_desc + " Combined",
                self.classes_,
                self.target_type
            )

            # Define the arrays for the new tree
            feature = [parent_a.feature[0]]
            # Use a threshold between the two
            threshold = [(parent_a.threshold[0] + parent_b.threshold[0]) / 2.0]

            start_right_tree_parent1 = parent_a.children_right[0]
            start_right_tree_parent2 = parent_b.children_right[0]
            feature.extend(parent_a.feature[1:start_right_tree_parent1])
            threshold.extend(parent_a.threshold[1:start_right_tree_parent1])
            children_left = list(parent_a.children_left[:start_right_tree_parent1].copy())
            children_right = list(parent_a.children_right[:start_right_tree_parent1].copy())

            offset = start_right_tree_parent2 - start_right_tree_parent1
            for node_idx in range(start_right_tree_parent2, len(parent_b.children_right)):
                feature.append(parent_b.feature[node_idx])
                threshold.append(parent_b.threshold[node_idx])
                children_left.append(parent_b.children_left[node_idx] - offset)
                children_right.append(parent_b.children_right[node_idx] - offset)

            idt.copy_from_values(feature, threshold, children_left, children_right)
            if self.target_type == 'classification':
                idt.count_training_records(x, y)
            else:
                idt.find_means(x, y)
            idt_list.append(idt)

        x = pd.DataFrame(x)
        y = pd.Series(y)
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.column_names = x.columns
        self.target_type = 'classification'
        if is_numeric_dtype(y) and y.nunique() > 20:
            self.target_type = 'regression'
        if self.target_type == 'classification':
            self.classes_ = sorted(y.unique())
        idt_list = []

        # Create the original set of decision trees based on bootstrap samples of the data. These may be similar to
        # each other.
        for i in range(10):
            x_sample = x.sample(n=len(x))
            y_sample = y.loc[x_sample.index]
            if self.target_type == 'classification':
                dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state + i)
            else:
                dt = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + i)
            dt.fit(x_sample, y_sample)
            idt = InternalDecisionTree("Original", self.classes_, self.target_type)
            idt.copy_from_dt(dt)
            idt_list.append(idt)
        idt_scores = get_dt_scores()

        for iteration_idx in range(self.max_iterations):
            if self.verbose:
                print()
                print(f"Iteration: {iteration_idx + 1}")

            top_scores_idxs = np.argsort(idt_scores)[::-1]

            # Take the top 10 trees in the current list and mutate them
            if self.allow_mutate:
                if self.n_jobs != 1:
                    # Execute in parallel
                    process_arr = []
                    with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                        for i in top_scores_idxs[:10]:
                            parent_idt = idt_list[i]
                            f = executor.submit(mutate_tree, parent_idt, x, y, self.classes_, self.target_type)
                            process_arr.append(f)
                        for f in concurrent.futures.as_completed(process_arr):
                            idt_list.extend(f.result())
                else:
                    # Execute in series
                    for i in top_scores_idxs[:10]:
                        parent_idt = idt_list[i]
                        ret_list = mutate_tree(parent_idt, x, y, self.classes_, self.target_type)
                        idt_list.extend(ret_list)

            # Take the pairs of trees with common nodes and combine them. This can take any pair from the current
            # top 20 (or 10 if this is the first iteration)
            if self.allow_combine:
                for i in range(len(top_scores_idxs)):
                    for j in range(i + 1, len(top_scores_idxs)):
                        idt_idx_1 = top_scores_idxs[i]
                        idt_idx_2 = top_scores_idxs[j]
                        parent_idt_1 = idt_list[idt_idx_1]
                        parent_idt_2 = idt_list[idt_idx_2]

                        # future: also allow starting lower in the tree; this checks just if the root nodes match
                        if parent_idt_1.feature[0] == parent_idt_2.feature[0]:
                            # Take the left child of parent 1 and the right child of parent 2
                            combine_trees(parent_idt_1, parent_idt_2)

                            # Take the right child of parent 1 and the left child of parent 2
                            combine_trees(parent_idt_2, parent_idt_1)

            # Generate more random trees. Each iteration, we generate less random trees and focus more on mutating the
            # best of the existing trees.
            num_random = 100 // (iteration_idx + 1)
            for i in range(num_random):
                # Take a random sample, based on a random sample size. For the sample size, we use a log distribution
                # and allow samples between 128 and 2 times the full data size.
                sample_size_log = np.random.uniform(low=7, high=np.log2(len(x) * 2))
                sample_size = int(math.pow(2, sample_size_log))
                x_sample = x.sample(n=sample_size, replace=True)
                y_sample = y.loc[x_sample.index]
                if self.target_type == 'classification':
                    dt = DecisionTreeClassifier(
                        max_depth=self.max_depth,
                        random_state=self.random_state + iteration_idx + i)
                else:
                    dt = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        random_state=self.random_state + iteration_idx + i)
                dt.fit(x_sample, y_sample)
                idt = InternalDecisionTree('Random', self.classes_, self.target_type)
                idt.copy_from_dt(dt)
                idt_list.append(idt)

            # Take the 20 with the top scores
            idt_scores = get_dt_scores()
            top_scores_idxs = np.argsort(idt_scores)[::-1][:20]
            idt_list = np.array(idt_list)[top_scores_idxs].tolist()
            idt_scores = np.array(idt_scores)[top_scores_idxs].tolist()
            if self.verbose:
                print("Top (training) scores so far:", [f"{x:.3f}" for x in idt_scores[:10]])
                sources_list = [idt.source_desc for idt in idt_list]
                if self.allow_mutate:
                    print("Number in top 20 based on mutation:", len([x for x in sources_list if "Modified" in x]))
                if self.allow_combine:
                    print("Number in top 20 based on combination:", len([x for x in sources_list if "Combined" in x]))

        top_scores_idxs = np.argsort(idt_scores)[::-1]
        idt_list = np.array(idt_list)[top_scores_idxs].tolist()
        self.internal_dt = idt_list[0]

    def continue_fitting(self, allow_mutate, allow_combine, n_iterations):
        """
        This may be called after fitting in order to execute additional iterations.

        :param allow_mutate: bool
        :param allow_combine: bool
        :param n_iterations: int
        :return: None
        """
        print("Unimplemented")

    def predict(self, x):
        """
        :param x:
        :return:
        """
        return self.internal_dt.predict(x)

    def export_tree(self):
        """
        Draws a representation of the tree, in a similar way as scikit-learn's decision tree.
        :return: None
        """

        # Get the depth of each node
        depths_arr = [-1]*len(self.internal_dt.feature)
        depths_arr[0] = 0
        changed_any = True
        while changed_any:
            changed_any = False
            for i in range(len(depths_arr)):
                if depths_arr[i] > -1:
                    left_child_idx = self.internal_dt.children_left[i]
                    right_child_idx = self.internal_dt.children_right[i]
                    if (left_child_idx > 0) and (depths_arr[left_child_idx] == -1):
                        depths_arr[left_child_idx] = depths_arr[i] + 1
                        changed_any = True
                    if (right_child_idx > 0) and (depths_arr[right_child_idx] == -1):
                        depths_arr[right_child_idx] = depths_arr[i] + 1
                        changed_any = True

        # Create a line for each node
        lines_arr = []
        for cur_node_idx in range(len(self.internal_dt.feature)):
            dots = "| ".join([""]*(depths_arr[cur_node_idx] + 1))
            if self.internal_dt.feature[cur_node_idx] >= 0:
                feature = self.internal_dt.feature[cur_node_idx]
                threshold = self.internal_dt.threshold[cur_node_idx]
                lines_arr.append(f"{dots}IF {self.column_names[feature]} < {threshold:.4f}")
            else:
                lines_arr.append(f"{dots}{self.internal_dt.node_prediction[cur_node_idx]}")

        # Create a map from the node ids to the lines ids
        node_id_to_line_id_map = {x:x for x in range(len(self.internal_dt.feature))}

        # Create a line for the right side of each condition
        for cur_node_idx in range(len(self.internal_dt.feature)-1, -1, -1):
            if self.internal_dt.feature[cur_node_idx] >= 0:
                dots = "| ".join([""]*(depths_arr[cur_node_idx] + 1))
                feature = self.internal_dt.feature[cur_node_idx]
                threshold = self.internal_dt.threshold[cur_node_idx]
                node_idx = self.internal_dt.children_right[cur_node_idx]
                lines_idx = node_id_to_line_id_map[node_idx]
                lines_arr.insert(lines_idx,
                                 f"{dots}ELSE {self.column_names[feature]} > {threshold:.4f}")
                for i in range(node_idx, len(self.internal_dt.feature)):
                    node_id_to_line_id_map[i] += 1

        # Render each line
        print()
        for line in lines_arr:
            print(line)


def mutate_tree(parent_idt, x, y, classes_, target_type):
    # Create 50 variations of each, where we select 5 nodes randomly and, for each, 10 random new
    # split points
    idt_list = []
    for j in range(5):  # Loop through each node that's modified
        n_nodes_parent = len(parent_idt.feature)
        while True:
            node_idx = np.random.choice(n_nodes_parent)
            if parent_idt.feature[node_idx] >= 0:
                break
        feature_idx = parent_idt.feature[node_idx]

        for k in range(10):  # loop through each new threshold
            idt = InternalDecisionTree(parent_idt.source_desc + " - Modified Threshold", classes_, target_type)
            new_threshold = parent_idt.threshold.copy()
            feat_name = x.columns[feature_idx]
            new_threshold[node_idx] = np.random.uniform(low=x[feat_name].min(), high=x[feat_name].max())

            idt.copy_from_values(
                feature=parent_idt.feature,
                threshold=new_threshold,
                children_left=parent_idt.children_left,
                children_right=parent_idt.children_right
            )
            if target_type == 'classification':
                idt.count_training_records(x, y)
            else:
                idt.find_means(x, y)
            idt_list.append(idt)
    return idt_list


def get_scores_classification(x, y, idt_list):
    return [f1_score(y, idt.predict(x), average='macro') for idt in idt_list]


def get_scores_regression(x, y, idt_list):
    return [r2_score(y, idt.predict(x)) for idt in idt_list]
