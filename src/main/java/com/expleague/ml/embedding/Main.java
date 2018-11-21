package com.expleague.ml.embedding;


import com.expleague.ml.embedding.exceptions.Word2VecUsageException;
import com.expleague.ml.embedding.quality_metrics.impl.ArithmeticMetric;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.List;

public class Main {
    
    //private static String INPUT_DATA = "data/corpuses/text8";
    //private static String INPUT_MODEL = "data/model_functions/hobbit/glove";

    public static void main(String[] args) throws IOException {
        Word2Vec word2Vec = new Word2Vec();
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        final int argsNumber = args.length;
        for (int i = 0; i < argsNumber; i++) {
            switch(args[i]) {
                case "-c":
                    if (i + 2 >= argsNumber)
                        throw new RuntimeException("Please, provide path to input data" +
                                "and directory where to save resulting model.");
                    String inputData = args[i + 1];
                    String modelPath = args[i + 2];
                    i += 2;
                    checkFile(inputData);
                    checkDirectory(modelPath);
                    prepareForTrain(word2Vec, modelTrainer, inputData);
                    train(word2Vec, modelTrainer, inputData, modelPath);
                    break;
                case "-lt":
                    if (i + 2 >= argsNumber)
                        throw new RuntimeException("Please, provide path to input data" +
                                "and directory with the model.");
                    inputData = args[i + 1];
                    modelPath = args[i + 2];
                    i += 2;
                    checkFile(inputData);
                    checkDirectory(modelPath);
                    loadModel(word2Vec, modelPath);
                    train(word2Vec, modelTrainer, inputData, modelPath);
                    break;
                case "--test":
                    if (i + 2 >= argsNumber)
                        throw new RuntimeException("Please, provide path to directory with the model.");
                    modelPath = args[i + 1];
                    checkDirectory(modelPath);
                    loadModel(word2Vec, modelPath);
                    String mode = args[i + 2];

                    if (mode.equals("-a")) {
                        if (i + 3 >= argsNumber)
                            throw new RuntimeException("Please, provide path to the file with metrics.");
                        String metricsData = args[i + 3];
                        i += 3;
                        checkFile(metricsData);
                        testArithmetic(word2Vec, metricsData);
                    } else if (mode.equals("-c")){
                        i += 2;
                        testClosest(word2Vec);
                    }
                    break;
                case "-h":
                case "--help":
                    System.out.println("[-c input/data/path final/model/directory] to create a new model and train it\n" +
                            "[-lt input/data/path existing/model/path] to load existing model and train it\n" +
                            "[--test existing/model/path [-c | -a metrics/data/path]] to load existing model and test it\n" +
                            "\t-a for arithmetics\n\t-c for top 5 closest\n");

            }
        }
    }

    private static void checkFile(String filePath) {
        File file = new File(filePath);
        if (!file.exists())
            throw new RuntimeException(filePath + " file doesn't exist.");
        if (!file.isFile())
            throw new RuntimeException(filePath + " is not a file.");
    }

    private static void checkDirectory(String dirPath) {
        File file = new File(dirPath);
        if (!file.exists())
            throw new RuntimeException(dirPath + " directory doesn't exist.");
        if (!file.isDirectory())
            throw new RuntimeException(dirPath + " is not a directory.");
    }

    private static void prepareForTrain(Word2Vec word2Vec, Word2Vec.ModelTrainer modelTrainer, String inputData) {
        modelTrainer.buildVocab(inputData);
        System.out.println("Vocabulary words_size: " + word2Vec.vocabSize());
    }

    private static void loadModel(Word2Vec word2Vec, String modelPath) throws IOException {
        word2Vec.loadModel(modelPath);
    }

    private static void train(Word2Vec word2Vec, Word2Vec.ModelTrainer modelTrainer,
                              String inputData, String modelPath) throws IOException {
        ModelParameters modelParameters = (new ModelParameters.Builder(inputData)).build();
        modelTrainer.trainModel(modelParameters);
        word2Vec.saveModel(modelPath);
    }

    private static void testArithmetic(Word2Vec word2Vec, String metricData) {
        Model model = word2Vec.getModel();
        System.out.println("Model likelihood = " + model.countLikelihood());
        ArithmeticMetric metric = new ArithmeticMetric(model);
        metric.measure(metricData, "data/tests/metric_result.txt");
    }

    private static void testClosest(Word2Vec word2Vec) throws IOException {
        Model model = word2Vec.getModel();
        System.out.println("Model likelihood = " + model.countLikelihood());
        String input;
        LineNumberReader lnr = new LineNumberReader(new InputStreamReader(System.in));
        while ((input = lnr.readLine()) != null) {
            try {
                List<String> result = model.getClosestWords(input, 5);
                for (String word : result)
                    System.out.println("\t|" + word);
            } catch (Word2VecUsageException e) {
                System.out.println("No such word");
            }
        }
    }
}
