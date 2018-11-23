package com.expleague.ml.embedding;


import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.ml.embedding.exceptions.Word2VecUsageException;
import com.expleague.ml.embedding.quality_metrics.impl.ArithmeticMetric;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.List;
import java.util.stream.IntStream;

public class Main {

    public static void main(String[] args) throws IOException {
        Word2Vec word2Vec = new Word2Vec();
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        final int argsNumber = args.length;
        for (int i = 0; i < argsNumber; i++) {
            switch(args[i]) {
                case "-c":
                    if (i + 3 >= argsNumber)
                        throw new RuntimeException("Please, provide path to input data " +
                                "and directory where to save resulting model.");
                    String inputData = args[i + 2];
                    String modelPath = args[i + 3];
                    String mode = args[i + 1];
                    i += 3;
                    checkFile(inputData);
                    checkDirectory(modelPath);
                    prepareForTrain(word2Vec, modelTrainer, inputData);
                    train(word2Vec, modelTrainer, inputData, modelPath, mode);
                    break;
                case "-lt":
                    if (i + 2 >= argsNumber)
                        throw new RuntimeException("Please, provide path to input data " +
                                "and directory with the model.");
                    inputData = args[i + 1];
                    modelPath = args[i + 2];
                    i += 2;
                    checkFile(inputData);
                    checkDirectory(modelPath);
                    loadModel(word2Vec, modelPath);
                    train(word2Vec, modelTrainer, inputData, modelPath, null);
                    break;
                case "--test":
                    if (i + 2 >= argsNumber)
                        throw new RuntimeException("Please, provide path to directory with the model.");
                    modelPath = args[i + 1];
                    checkDirectory(modelPath);
                    System.out.println("Loading started");
                    loadModel(word2Vec, modelPath);
                    System.out.println("Loading finished");
                    mode = args[i + 2];
                    if (mode.equals("-a")) {
                        if (i + 4 >= argsNumber)
                            throw new RuntimeException("Please, provide path to the file with metrics and to filename where to store results.");
                        String metricsNames = args[i + 3];
                        String resultDir = args[i + 4];
                        i += 4;
                        checkFile(metricsNames);
                        checkDirectory(resultDir);
                        testArithmetic(word2Vec, metricsNames, resultDir);
                    } else if (mode.equals("-c")){
                        i += 2;
                        testClosest(word2Vec);
                    }
                    break;
                case "-h":
                case "--help":
                    System.out.println("[-c [-g | -d] input/data/path final/model/directory] to create a new model and train it\n" +
                            "\t-g for GLOVE\n\t-d for DECOMPOSITION\n" +
                            "[-lt input/data/path existing/model/path] to load existing model and train it\n" +
                            "[--test existing/model/path [-c | -a metrics_names/file/path metrics/result/dir] to load existing model and test it\n" +
                            "\t-a <FILE> <DIR> for arithmetics, names of all_metrics_files.txt files with 4-word metrics in file, " +
                            "all_metrics_files.txt results will be stored in directory\n\t-c for top 5 closest\n");

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
                              String inputData, String modelPath, String mode) throws IOException {
        ModelParameters modelParameters = (new ModelParameters.Builder(inputData)).setModelName(mode).build();
        modelTrainer.trainModel(modelParameters);
        word2Vec.saveModel(modelPath);
    }

    private static void testArithmetic(Word2Vec word2Vec, String metricNames, String resultDir) {
        Model model = word2Vec.getModel();
        System.out.println("Model likelihood = " + model.countLikelihood());
        ArithmeticMetric metric = new ArithmeticMetric(model);
        metric.measure(metricNames, resultDir);
    }

    private static void testClosest(Word2Vec word2Vec) throws IOException {
        Model model = word2Vec.getModel();
        System.out.println("Model likelihood = " + model.countLikelihood());
        String input;
        LineNumberReader lnr = new LineNumberReader(new InputStreamReader(System.in));
        while ((input = lnr.readLine()) != null) {
            try {
                List<String> result = model.getClosestWords(input, 5);
                System.out.println(model.getVectorByWord(input));
                for (String word : result)
                    System.out.println("\t|" + word);
            } catch (Word2VecUsageException e) {
                System.out.println("No such word");
            }
        }
    }
}
