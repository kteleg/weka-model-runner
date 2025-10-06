package com.example.weka;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.xgboost.XGBoost;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.RBFNetwork;

import java.io.File;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class WekaModelRunner {

    private static final Map<String, String> datasets = new HashMap<>();
    static {
        datasets.put("tennis", "/tennis.csv");
        datasets.put("iris", "/IRIS.csv");
        datasets.put("wine", "/winequality-red.csv");   
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java WekaModelRunner <model> <datasetKeyOrPath> <targetColIndex> <task>");
            System.out.println("Models: J48, RandomForest, MLP, SGD, RBF, XGBoost");
            System.out.println("Datasets: " + datasets.keySet() + " or local path");
            System.out.println("Target Index: starting from 0");
            System.out.println("Task: reg (regression) or cls (classification)");
            return;
        }

        String modelName = args[0];
        String datasetKeyOrPath = args[1];
        int targetIndex;
       
        String taskType = args[3].toLowerCase();
        
        boolean isRegression = taskType.equals("reg");

        // Resolve dataset path
        String datasetPath;
        Instances dataset;
        if (datasets.containsKey(datasetKeyOrPath)) {
            datasetPath = datasets.get(datasetKeyOrPath);
            CSVLoader loader = new CSVLoader();
            InputStream is = WekaModelRunner.class.getResourceAsStream(datasetPath);
            if (is == null) {
                throw new IllegalArgumentException("Dataset not found in resources: " + datasetPath);
            }
            loader.setSource(is);

            dataset = loader.getDataSet();
            
        } else {
            File customFile = new File(datasetKeyOrPath);
            if (!customFile.exists()) {
                System.err.println("Dataset not found: " + datasetKeyOrPath);
                System.exit(1);
                return;
            }
            datasetPath = customFile.getAbsolutePath();
            DataSource source = new DataSource(datasetPath);
            dataset = source.getDataSet();
        }
       
        printDatasetHead(dataset, 5);
        
        if (args[2].equals("-1")) {
            targetIndex = dataset.numAttributes() - 1; // last column
        } else {
            targetIndex = Integer.parseInt(args[2]);
        }
        
        // Set class index
        dataset.setClassIndex(targetIndex);
        Attribute targetAttr = dataset.classAttribute();

        System.out.println("=== Target column info ===");
        System.out.println("Name: " + targetAttr.name());
        System.out.println("Type: " + (targetAttr.isNumeric() ? "Numeric" : "Nominal"));

        if (!isRegression) {
            System.out.println("Converting target column to nominal using NumericToNominal() for classification");
	        // Convert target column to nominal if classification
	        NumericToNominal convert = new NumericToNominal();
	        convert.setAttributeIndices("" + (targetIndex + 1)); // Weka indices are 1-based
	        convert.setInputFormat(dataset);
	        dataset = Filter.useFilter(dataset, convert);
        }


        // For neural nets and some models, convert nominal attributes to binary
        if (modelName.equalsIgnoreCase("MLP") || modelName.equalsIgnoreCase("SGD") ||
            modelName.equalsIgnoreCase("RBF")){
            System.out.println("The model internally converts features from nominal to binary (one-hot encoding) using NominalToBinary()");
           
        }
        else if (modelName.equalsIgnoreCase("XGBoost")) {
            System.out.println("Converting features from nominal to binary (one-hot encoding) using NominalToBinary()");
	    	NominalToBinary nominalToBinary = new NominalToBinary();
	        nominalToBinary.setInputFormat(dataset);
	        dataset = Filter.useFilter(dataset, nominalToBinary);
        }
        else {
            System.out.println(modelName + " supports nominal attributes directly. No conversion needed.");

        }
        
        // Shuffle and split dataset (80% train, 20% test)
        dataset.randomize(new Random(42));
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        Instances trainSet = new Instances(dataset, 0, trainSize);
        Instances testSet = new Instances(dataset, trainSize, testSize);

        System.out.println("=== Training set preview ===");
        printDatasetHead(trainSet, 5);
        System.out.println("=== Testing set preview ===");
        printDatasetHead(testSet, 5);

        // Instantiate the selected classifier
        Classifier model = getModel(modelName);
        if (model == null) {
            System.out.println("Unknown model: " + modelName);
            return;
        }

        // Train the model
        model.buildClassifier(trainSet);

        System.out.println("=== Trained " + modelName + " on " + trainSet.numInstances() + " instances ===");
        System.out.println(model);
        
        evaluate(isRegression, testSet, model);
    }

	private static void evaluate(boolean isRegression, Instances dataset, Classifier model) throws Exception {
		Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(model, dataset);

        System.out.println(eval.toSummaryString("\n=== Model Evaluation ===\n", false));

        if (!isRegression) {
            System.out.println("Confusion Matrix:");
            for (double[] row : eval.confusionMatrix()) {
                System.out.println(Arrays.toString(row));
            }
        } else {
            System.out.printf("Correlation coefficient: %.4f\n", eval.correlationCoefficient());
            System.out.printf("Mean absolute error: %.4f\n", eval.meanAbsoluteError());
            System.out.printf("Root mean squared error: %.4f\n", eval.rootMeanSquaredError());
        }
	}

    private static Classifier getModel(String name) throws Exception {
        switch (name.toUpperCase()) {
            case "J48":
                return new J48();
            case "RANDOMFOREST":
                return new RandomForest();
            case "MLP":
                MultilayerPerceptron mlp = new MultilayerPerceptron();
                mlp.setHiddenLayers("3");
                mlp.setTrainingTime(500);
                return mlp;
            case "SGD":
                SGD sgd = new SGD();
                sgd.setLossFunction(new SelectedTag(SGD.HINGE, SGD.TAGS_SELECTION));
                sgd.setEpochs(500);
                return sgd;
            case "RBF":
                return new RBFNetwork();
            case "XGBOOST":
                return new XGBoost();
            default:
                return null;
        }
    }
    
    private static void printDatasetHead(Instances dataset, int n) {
        System.out.println("=== Dataset preview (first " + n + " rows) ===");
        int rows = Math.min(n, dataset.numInstances());
        for (int i = 0; i < dataset.numAttributes(); i++) {
            System.out.print(dataset.attribute(i).name() + "\t");
        }
        System.out.println();
        for (int i = 0; i < rows; i++) {
            System.out.println(dataset.instance(i));
        }
    }

}

