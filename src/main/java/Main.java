import word2vec.ModelParameters;
import word2vec.Word2Vec;
import word2vec.exceptions.Word2VecUsageException;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.Scanner;

public class Main {
    
    //private static String INPUT_DATA = "data/corpuses/text8";
    //private static String INPUT_MODEL = "data/models/hobbit/glove";

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
                                "and directory where to save resulting model.");
                    inputData = args[i + 1];
                    modelPath = args[i + 2];
                    i += 2;
                    checkFile(inputData);
                    checkDirectory(modelPath);
                    loadModel(word2Vec);
                    train(word2Vec, modelTrainer, inputData, modelPath);
                    break;
                case "-h":
                case "--help":
                    System.out.println("[-c input/data/path final/model/directory] to create a new model and train it\n" +
                            "[-lt input/data/path existing/model/path] to load existing model and train it\n");

            }
        }

        // Test with closer-further metrics
        /*word2Vec.loadModel("data/models/hobbit");
        Word2Vec.Model model = word2Vec.getModel();
        CloserFurtherMetric metric = new CloserFurtherMetric(model);
        metric.measure("data/tests/hobbit/hobbit_logic.txt","data/tests/hobbit/hobbit_logic_result.txt");*/
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
        System.out.println("Vocabulary size: " + word2Vec.vocabSize());
    }

    private static void loadModel(Word2Vec word2Vec) throws IOException {
        word2Vec.loadModel("data/models/hobbit/glove");
    }

    private static void train(Word2Vec word2Vec, Word2Vec.ModelTrainer modelTrainer,
                              String inputData, String modelPath) throws IOException {
        ModelParameters modelParameters = (new ModelParameters.Builder(inputData)).setModelName("GLOVE").build();
        modelTrainer.trainModel(modelParameters);
        word2Vec.saveModel(modelPath);
    }

    private static void testModel(Word2Vec word2Vec) throws IOException {
        Word2Vec.Model model = word2Vec.getModel();
        System.out.println(model.countLikelihood());
        String input;
        LineNumberReader lnr = new LineNumberReader(new InputStreamReader(System.in));
        while ((input = lnr.readLine()) != null) {
            try {
                System.out.println(model.getClosest(input));
            } catch (Word2VecUsageException e) {
                System.out.println("No such word");
            }
        }
    }
}
