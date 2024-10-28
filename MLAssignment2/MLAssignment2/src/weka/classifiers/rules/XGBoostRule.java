package weka.classifiers.rules;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Collectors;

/**
 * Basic WEKA classifiers (and this includes learning algorithms that build rules!)
 * should simply extend AbstractClassifier but this classifier is also randomizable.
 */

public class XGBoostRule extends RandomizableClassifier implements WeightedInstancesHandler, AdditionalMeasureProducer {

    /**
     * Provides an enumeration consisting of a single element: "measureNumRules".
     */

    public Enumeration<String> enumerateMeasures() {
        String[] measures = {"measureNumRules"};
        return Collections.enumeration(Arrays.asList(measures));
    }
    /**
     * Provides the number of leaves for "measureNumRules" and throws an exception for other arguments. - in this case it will give rule
     */
    public double getMeasure(String measureName) throws IllegalArgumentException {
        if (measureName.equals("measureNumRules")) {
            return getNumLeaves(rootNode);
        } else {
            throw new IllegalArgumentException("Measure " + measureName + " not supported.");
        }
    }
    /**
     * The hyperparameters for an XGBoost Rule except for
     * The max_length(default as 6) is a hyperparameter in XGBoostRule that limits the maximum number of conditions a rule can have and
     * preventing the rule from becoming too complex and overfitting the data.
     */
    private double eta = 0.3;

    @OptionMetadata(displayName = "eta", description = "eta",
            commandLineParamName = "eta", commandLineParamSynopsis = "-eta <double>", displayOrder = 1)
    public void setEta(double e) { eta = e; }

    public double getEta() { return eta; }

    private double lambda = 1.0;

    @OptionMetadata(displayName = "lambda", description = "lambda",
            commandLineParamName = "lambda", commandLineParamSynopsis = "-lambda <double>", displayOrder = 2)
    public void setLambda(double l) { lambda = l; }

    public double getLambda() { return lambda; }

    private double gamma = 1.0;

    @OptionMetadata(displayName = "gamma", description = "gamma",
            commandLineParamName = "gamma", commandLineParamSynopsis = "-gamma <double>", displayOrder = 3)
    public void setGamma(double l) { gamma = l; }

    public double getGamma() { return gamma; }

    private double subsample = 0.5;

    @OptionMetadata(displayName = "subsample", description = "subsample",
            commandLineParamName = "subsample", commandLineParamSynopsis = "-subsample <double>", displayOrder = 4)
    public void setSubsample(double s) { subsample = s; }

    public double getSubsample() { return subsample; }

    private double colsample_bynode = 1.0;

    @OptionMetadata(displayName = "colsample_bynode", description = "colsample_bynode",
            commandLineParamName = "colsample_bynode", commandLineParamSynopsis = "-colsample_bynode <double>", displayOrder = 5)
    public void setColSampleByNode(double c) { colsample_bynode = c; }

    public double getColSampleByNode() { return colsample_bynode; }

    private int max_length = 6;

    @OptionMetadata(displayName = "max_length", description = "max_length",
            commandLineParamName = "max_length", commandLineParamSynopsis = "-max_length <int>", displayOrder = 6)
    public void setMaxLength(int m) { max_length = m; }

    public int getMaxLength() { return max_length; }

    private double min_child_weight = 1.0;

    @OptionMetadata(displayName = "min_child_weight", description = "min_child_weight",
            commandLineParamName = "min_child_weight", commandLineParamSynopsis = "-min_child_weight <double>", displayOrder = 7)
    public void setMinChildWeight(double w) { min_child_weight = w; }

    public double getMinChildWeight() { return min_child_weight; }

    private interface Node extends Serializable { }


    // Represents an internal node in the decision rule, holding the attribute, split point, split quality and a reference to the child node.
    // The booleanTest determines if the split is >= (true) or < (false).

    private record InternalNode(Attribute attribute, double splitPoint, double splitQuality, boolean booleanTest, Node child)
            implements Node, Serializable { }


    // Represents a leaf node in the rule which holds the final prediction value.
    // Leaf nodes indicate the end of a rule(consequent), providing the prediction for the instances that meet the rule's conditions.

    private record LeafNode(double prediction) implements Node, Serializable { }

    /**
     * The root node of the decision rule.
     */
    private Node rootNode = null;

    /**
     * The training instances.
     */
    private Instances data;
    /**
     * Random number generator to be used for subsampling rows and columns.
     */
    Random random;
    /**
     * A class for objects that hold a split specification, including the quality of the split,
     * boolean test which will hold '<' false by default for rule growing, split point
     * and added Serializable implementation as required for WEKA deep copy
     */

    private class SplitSpecification implements Serializable {
        private final Attribute attribute;
        private double splitPoint;
        private double splitQuality;
        private boolean booleanTest;

        private SplitSpecification(Attribute attribute, double splitQuality, double splitPoint, boolean booleanTest) {
            this.attribute = attribute;
            this.splitQuality = splitQuality;
            this.splitPoint = splitPoint;
            this.booleanTest = booleanTest;
        }
    }

    /**
     * A class for objects that contain the sufficient statistics required to measure split quality
     * and added Serializable implementation as required for WEKA deep copy
     */
    private class SufficientStatistics implements Serializable {
        private double sumOfNegativeGradients = 0.0;
        private double sumOfHessians = 0.0;

        private SufficientStatistics(double sumOfNegativeGradients, double sumOfHessians) {
            this.sumOfNegativeGradients = sumOfNegativeGradients;
            this.sumOfHessians = sumOfHessians;
        }

        private void updateStats(double negativeGradient, double hessian, boolean add) {
            sumOfNegativeGradients = (add) ? sumOfNegativeGradients + negativeGradient
                    : sumOfNegativeGradients - negativeGradient;
            sumOfHessians = (add) ? sumOfHessians + hessian
                    : sumOfHessians - hessian;
        }
    }

    /**
     * Below code calculates the split quality based on the sum of negative gradients and Hessians, penalized by lambda and gamma (regularisation).
     * T is the length or depth of the rule, and gamma controls the penalty for longer or more complex rules.
     * The formula is maximizing the gain while minimizing overfitting by applying regularisation.
     */
    private double splitQuality(SufficientStatistics stats, int T) {
        return ((0.5 * ((stats.sumOfNegativeGradients * stats.sumOfNegativeGradients) / (stats.sumOfHessians + lambda))) - (gamma * T));
    }
    /**
     * Below code finds the best test condition (split point) for a given attribute by evaluating both left and right splits.
     * Firstly, the left and right statistics (gradients and Hessians) are initialised for the split decision.
     *  Then, It loops over all instances calculating whether the current instance value for the attribute is greater than the previous value.
     * Then, for each split, it checks if the left and right conditions are valid based on the minimum child weight as given.
     * Also,it calculates split quality for both left (<) and right (>=) splits and compares them to determine the best split for rule grow.
     * The split specification is updated with the best split based on the higher quality as per the assignment.
     * Finally, it returns the best split condition for the attribute with boolean test which false for < and true for >= for rule grow
     */
    private SplitSpecification findBestTestCondition(int[] indices, Attribute attribute, SufficientStatistics stats, int T) {
        var leftStats = new SufficientStatistics(0.0, 0.0);
        var rightStats = new SufficientStatistics(stats.sumOfNegativeGradients, stats.sumOfHessians);
        var splitSpecification = new SplitSpecification(attribute, 0, Double.NEGATIVE_INFINITY, false);
        var previousValue = Double.NEGATIVE_INFINITY;

        for (int i : Arrays.stream(Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x ->
                data.instance(x).value(attribute)).toArray())).map(x -> indices[x]).toArray()) {
            Instance instance = data.instance(i);

            if (instance.value(attribute) > previousValue) {

                // comment to handle the first iteration: only update previousValue and stats without performing split
                if (previousValue == Double.NEGATIVE_INFINITY) {
                    previousValue = instance.value(attribute);
                    leftStats.updateStats(instance.classValue(), instance.weight(), true);  // Move the first instance to the left
                    rightStats.updateStats(instance.classValue(), instance.weight(), false); // Update right stats
                    continue;  // Skip the rest of the loop for the first iteration
                }
                // Checking left and right conditions separately as we will grow in one direction the rule
                boolean validLeft = leftStats.sumOfHessians >= min_child_weight;
                boolean validRight = rightStats.sumOfHessians >= min_child_weight;

                if (leftStats.sumOfHessians != 0 && validLeft) {
                    var leftQuality = splitQuality(leftStats, T + 1); // Left (<) split if found better update the split specification
                    if (leftQuality > splitSpecification.splitQuality) {
                        splitSpecification.splitQuality = leftQuality;
                        splitSpecification.splitPoint = (instance.value(attribute) + previousValue) / 2.0;
                        splitSpecification.booleanTest = false;
                    }
                }

                if (rightStats.sumOfHessians != 0 && validRight) {
                    var rightQuality = splitQuality(rightStats, T + 1); // Right (>=) split if found better update the split specification
                    if (rightQuality > splitSpecification.splitQuality) {
                        splitSpecification.splitQuality = rightQuality;
                        splitSpecification.splitPoint = (instance.value(attribute) + previousValue) / 2.0;
                        splitSpecification.booleanTest = true;
                    }
                }
            }

            previousValue = instance.value(attribute);
            leftStats.updateStats(instance.classValue(), instance.weight(), true);
            rightStats.updateStats(instance.classValue(), instance.weight(), false);
        }

        return splitSpecification;
    }

    /**
     * This method creates a decision rule by recursively finding the best split point for the given data and growing the rule.
     * indices: The indices of the data points being considered.
     * T: The current depth or length of the rule (used to limit the rule length as per the assignment mentioned).
     * parentQuality: The quality of the parent node's split to compare whether to grow rule or not.
     */
    private Node makeRule(int[] indices, int T, double parentQuality) {
        var stats = new SufficientStatistics(0.0, 0.0);
        for (int i : indices) {
            stats.updateStats(data.instance(i).classValue(), data.instance(i).weight(), true);
        }

        // Stop if rule reaches max length or other constraints as mentioned in the assignment
        if (stats.sumOfHessians <= 0.0 || stats.sumOfHessians < min_child_weight || T >= max_length) {
            return new LeafNode(eta * stats.sumOfNegativeGradients / (stats.sumOfHessians + lambda));
        }

        var bestSplitSpecification = new SplitSpecification(null, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, false); // initially declare as false
        List<Integer> attributes = new ArrayList<>(this.data.numAttributes() - 1);

        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != this.data.classIndex()) {
                attributes.add(i);
            }
        }

        if (colsample_bynode < 1.0) {
            Collections.shuffle(attributes, random);
        }

        for (Integer index : attributes.subList(0, (int) (colsample_bynode * attributes.size()))) {
            var splitSpecification = findBestTestCondition(indices, data.attribute(index), stats, T);

            if (splitSpecification.splitQuality > bestSplitSpecification.splitQuality) {
                bestSplitSpecification = splitSpecification;
            }
        }

        // Stop if the split quality is not improving taking number close to zero and also consider the negative split doesn't happen
        if (bestSplitSpecification.splitQuality <= 1E-6 || bestSplitSpecification.splitQuality - parentQuality <= 1E-6) {
            return new LeafNode(eta * stats.sumOfNegativeGradients / (stats.sumOfHessians + lambda));
        }

        var leftSubset = new ArrayList<Integer>(indices.length);
        var rightSubset = new ArrayList<Integer>(indices.length);

        for (int i : indices) {
            if (data.instance(i).value(bestSplitSpecification.attribute) < bestSplitSpecification.splitPoint) {
                leftSubset.add(i);
            } else {
                rightSubset.add(i);
            }
        }
        // note instead of traversing in both left and right like tree creation, rule will be created by going on one side with best split quality like linked list
        if (bestSplitSpecification.booleanTest) {  // true means >=
            return new InternalNode(bestSplitSpecification.attribute, bestSplitSpecification.splitPoint, bestSplitSpecification.splitQuality, true,
                    makeRule(rightSubset.stream().mapToInt(Integer::intValue).toArray(), T + 1, bestSplitSpecification.splitQuality));
        } else { // false means <
            return new InternalNode(bestSplitSpecification.attribute, bestSplitSpecification.splitPoint, bestSplitSpecification.splitQuality, false,
                    makeRule(leftSubset.stream().mapToInt(Integer::intValue).toArray(), T + 1, bestSplitSpecification.splitQuality));
        }
    }

    /**
     * Returns the capabilities of the classifier: numeric predictors and numeric target.
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        return result;
    }
    /**
     * Builds the rule by calling the recursive makeRule(Instances) method.
     */
    public void buildClassifier(Instances trainingData) throws Exception {
        getCapabilities().testWithFail(trainingData);
        random = new Random(getSeed());
        this.data = new Instances(trainingData);
        if (subsample < 1.0) {
            this.data.randomize(random);
        }
        this.data = new Instances(this.data, 0, (int) (subsample * this.data.numInstances()));
        rootNode = makeRule(IntStream.range(0, this.data.numInstances()).toArray(), 0, Double.NEGATIVE_INFINITY);
        data = null;
        random = null;
    }

    /**
     * Recursive method for obtaining a prediction from the rule which is consequent attached to the node provided.
     */
    private double makePrediction(Node node, Instance instance) {
        if (node instanceof LeafNode) {
            return ((LeafNode) node).prediction;
        } else if (node instanceof InternalNode) {
            InternalNode internalNode = (InternalNode) node;
            if (internalNode.booleanTest && instance.value(internalNode.attribute) >= internalNode.splitPoint) {
                return makePrediction(internalNode.child, instance);
            } else if (!internalNode.booleanTest && instance.value(internalNode.attribute) < internalNode.splitPoint) {
                return makePrediction(internalNode.child, instance);
            } else{
                return 0.0; // no rule or test apply to instance
            }
        }

        return Utils.missingValue(); // This should never happen
    }
    /**
     * Provides a prediction for the current instance by calling the recursive makePrediction(Node, Instance) method if covered by the respective rule.
     */
    public double classifyInstance(Instance instance) {
        return makePrediction(rootNode, instance);
    }

    public int getNumLeaves(Node node) {
        if (node instanceof LeafNode) {
            return 1;    // just matches the XGBoostTree structure but it will return 1 as we're forming 1 rule per iteration so return 1, implemented just to cross-check rather than hard coding value
        } else {
            return getNumLeaves(((InternalNode)node).child);
        }
    }
    /**
     * Recursively produces the string representation of a branch in the rule growing like if then rule.
     */
    private void branchToString(StringBuffer sb, boolean direction, int level, InternalNode node, String ruleSoFar) {
        // Use the current node's booleanTest to determine the condition (false for < or true for >=)
        String condition = node.attribute.name() + (node.booleanTest ? " >= " : " < ") + Utils.doubleToString(node.splitPoint, getNumDecimalPlaces());
        String updatedRule = ruleSoFar.isEmpty() ? condition : ruleSoFar + " and " + condition;

        if (node.child instanceof InternalNode) {
            branchToString(sb, node.booleanTest, level + 1, (InternalNode) node.child, updatedRule);
        } else {
            toString(sb, level + 1, node.child, updatedRule);
        }
    }

    /**
     * Recursively produces a string representation of a subtree by calling the branchToString(StringBuffer, boolean ,int,
     * Node, string) method for both branches, unless we are at a leaf.
     */
    private void toString(StringBuffer sb, int level, Node node, String ruleSoFar) {
        if (node instanceof LeafNode) {
            if (ruleSoFar.isEmpty()) {
                sb.append("if true then " + ((LeafNode) node).prediction + "\n");
            } else {
                sb.append("if " + ruleSoFar + " then " + ((LeafNode) node).prediction + "\n");
            }
        } else {
            InternalNode internalNode = (InternalNode) node;
            // Pass the correct booleanTest from the current internal node
            branchToString(sb, internalNode.booleanTest, level, internalNode, ruleSoFar);
        }
    }

    /**
     * Returns a string representation of the tree by calling the recursive toString(StringBuffer, int, Node, string) method.
     */

    public String toString() {
        StringBuffer sb = new StringBuffer();
        toString(sb, 0, rootNode, "");
        return sb.toString();
    }

    public static void main(String[] options) {
        runClassifier(new XGBoostRule(), options);
    }
}
