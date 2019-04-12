package com.expleague.ml.embedding;

import com.expleague.ml.embedding.exceptions.Word2VecUsageException;
import com.expleague.ml.embedding.quality_metrics.impl.ArithmeticMetric;
import com.expleague.ml.embedding.quality_metrics.impl.CloserFurtherMetric;

import java.io.*;
import java.util.List;

public class Main {

  public static void main(String[] args) throws IOException {
    String input = "data/corpuses/text8";
    String res = "data/models/text8/shrinking/";
    String metricNames = "data/tests/text8/all_metrics_files.txt";
    String resultDec = "data/tests/text8/results_decomp/shrinking/";
    String resultGl = "data/tests/text8/results_glove/shrinking/";
    int[] glove = {150};
    int[] sym = {120};
    int[] skew = {10, 20};
    int[] iters = {25};

    for (int gl : glove) {
      for (int it : iters) {
        Word2Vec word2Vec = new Word2Vec();
        System.out.println(String.format("GLOVE-%d_it-%d started", gl, it));
        /*word2Vec.loadModel(res + String.format("GLOVE-%d_it-%d", gl, it), 1);
        Model model = word2Vec.getModel();
        ArithmeticMetric metric = new ArithmeticMetric(model);
        metric.measure(metricNames, resultGl + String.format("sz-%d_it-%d", gl, it));*/

        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.buildVocab(input);
        ModelParameters modelParameters = (new ModelParameters.Builder(input)).setModelName("GLOVE").
                setGloveVecSize(gl).setTrainingIters(it).build();
        modelTrainer.trainModel(modelParameters);
        word2Vec.saveModel(res + String.format("GLOVE-%d_it-%d", gl, it));
        System.out.println("\n\n");
      }
    }

    for (int sy : sym) {
      for (int sk : skew) {
        for (int it : iters) {
          Word2Vec word2Vec = new Word2Vec();
          System.out.println(String.format("DECOMP-%d-%d_it-%d", sy, sk, it));
          /*Word2Vec word2Vec = new Word2Vec();
          word2Vec.loadModel(res + String.format("DECOMP-%d-%d_it-%d", sy, sk, it), 1);
          Model model = word2Vec.getModel();
          ArithmeticMetric metric = new ArithmeticMetric(model);
          metric.measure(metricNames, resultDec + String.format("sz-%d-%d_it-%d", sy, sk, it));*/

          Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
          modelTrainer.buildVocab(input);
          ModelParameters modelParameters = (new ModelParameters.Builder(input)).setModelName("DECOMP").
                  setSkewSize(sk).setSymSize(sy).setTrainingIters(it).build();
          modelTrainer.trainModel(modelParameters);
          word2Vec.saveModel(res + String.format("DECOMP-%d-%d_it-%d", sy, sk, it));
          System.out.println("\n\n");
        }
      }
    }


    /*final int argsNumber = args.length;
    for (int i = 0; i < argsNumber; i++) {
      switch(args[i]) {
        case "-c":
          if (i + 3 >= argsNumber)
            throw new RuntimeException("Please, provide mode (GLOVE|DECOMP), path to input data and directory where to save resulting model.");
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
            throw new RuntimeException("Please, provide path to input data and directory with the model.");
          inputData = args[i + 1];
          modelPath = args[i + 2];
          i += 2;
          checkFile(inputData);
          checkDirectory(modelPath);
          loadModel(word2Vec, modelPath, 0);
          train(word2Vec, modelTrainer, inputData, modelPath, null);
          break;
        case "--test":
          if (i + 2 >= argsNumber)
            throw new RuntimeException("Please, provide path to directory with the model.");
          modelPath = args[i + 1];
          checkDirectory(modelPath);
          System.out.println("Loading started");
          loadModel(word2Vec, modelPath, 1);
          System.out.println("Loading finished");
          mode = args[i + 2];
          if (mode.equals("-a") | mode.equals("-cf")) {
            if (i + 4 >= argsNumber)
              throw new RuntimeException("Please, provide path to the file with metrics and to filename where to store results.");
            String metricsNames = args[i + 3];
            String resultDir = args[i + 4];
            i += 4;
            checkFile(metricsNames);
            checkDirectory(resultDir);
            if (mode.equals("-a"))
              testArithmetic(word2Vec, metricsNames, resultDir);
            else if (mode.equals("-cf"))
              testCloserFurter(word2Vec, metricsNames, resultDir);
          } else if (mode.equals("-c")){
            i += 2;
            testClosest(word2Vec);
          }
          break;
        case "-h":
        case "--help":
          System.out.println("[-c [GLOVE | DECOMP] input/data/path final/model/directory] to create a new model and train it\n" +
                  "\t-g for GLOVE\n\t-d for DECOMPOSITION\n" +
                  "[-lt input/data/path existing/model/path] to load existing model and train it\n" +
                  "[--test existing/model/path -c to load existing model and see top5 closes words\n" +
                  "[--test existing/model/path -cf metrics_names/file/path metrics/result/dir] to load existing model and test it on closer-further metrics\n" +
                  "[--test existing/model/path -a metrics_names/file/path metrics/result/dir] to load existing model and test it on arithmetic metrics\n" +
                  "\t-a <FILE> <DIR> for arithmetics, names of all_metrics_files.txt files with 4-word metrics in file, " +
                  "all_metrics_files.txt results will be stored in directory\n\t-c for top 5 closest\n");

      }
    }*/
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

  private static void loadModel(Word2Vec word2Vec, String modelPath, int mode) throws IOException {
    word2Vec.loadModel(modelPath, mode);
  }

  private static void train(Word2Vec word2Vec, Word2Vec.ModelTrainer modelTrainer,
                            String inputData, String modelPath, String mode) throws IOException {
    ModelParameters modelParameters = (new ModelParameters.Builder(inputData)).setModelName(mode).build();
    modelTrainer.trainModel(modelParameters);
    word2Vec.saveModel(modelPath);
  }

  private static void testArithmetic(Word2Vec word2Vec, String metricNames, String resultDir) {
    Model model = word2Vec.getModel();
    ArithmeticMetric metric = new ArithmeticMetric(model);
    metric.measure(metricNames, resultDir);
  }

  private static void testCloserFurter(Word2Vec word2Vec, String metricNames, String resultDir) {
    Model model = word2Vec.getModel();
    CloserFurtherMetric metric = new CloserFurtherMetric(model);
    metric.measure(metricNames, resultDir);
  }

  private static void testClosest(Word2Vec word2Vec) throws IOException {
    Model model = word2Vec.getModel();
    String input;
    LineNumberReader lnr = new LineNumberReader(new InputStreamReader(System.in));
    System.out.println("Enter your word.");
    while ((input = lnr.readLine()) != null) {
      try {
        System.out.println(model.getIndexByWord(input));
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
