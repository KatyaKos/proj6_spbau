import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import jdk.nashorn.internal.runtime.WithObject;
import word2vec.ModelParameters;
import word2vec.Word2Vec;
import word2vec.exceptions.Word2VecUsageException;
import word2vec.quality_metrics.QualityMetric;
import word2vec.quality_metrics.impl.CloserFurtherMetric;

import java.io.IOException;
import java.io.*;
import java.util.Scanner;

public class Main {
    
    private static String INPUT_DATA = "data/corpuses/hobbit.txt";
    private static String INPUT_MODEL = "data/models/hobbit/glove";

    public static void main(String[] args) throws IOException {
        Word2Vec word2Vec = new Word2Vec();
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();

        System.out.println("If you want to create new model, press 1.\n" +
                "If you want to load model and continue training, press 2.\n" +
                "If you want to load model and test it, press 3.");

        final int task = (new Scanner(System.in)).nextInt();
        switch (task){
            case 1:
                prepareForTrain(word2Vec, modelTrainer);
                train(word2Vec, modelTrainer);
                break;
            case 2:
                loadModel(word2Vec);
                train(word2Vec, modelTrainer);
                break;
            case 3:
                loadModel(word2Vec);
                testModel(word2Vec);
                break;
        }

        // Test with closer-further metrics
        /*word2Vec.loadModel("data/models/hobbit");
        Word2Vec.Model model = word2Vec.getModel();
        CloserFurtherMetric metric = new CloserFurtherMetric(model);
        metric.measure("data/tests/hobbit/hobbit_logic.txt","data/tests/hobbit/hobbit_logic_result.txt");*/
    }

    private static void prepareForTrain(Word2Vec word2Vec, Word2Vec.ModelTrainer modelTrainer) {
        modelTrainer.buildVocab(INPUT_DATA);
        System.out.println(word2Vec.vocabSize());
    }

    private static void loadModel(Word2Vec word2Vec) throws IOException {
        word2Vec.loadModel("data/models/hobbit/glove");
    }

    private static void train(Word2Vec word2Vec, Word2Vec.ModelTrainer modelTrainer) throws IOException {
        ModelParameters modelParameters = (new ModelParameters.Builder(INPUT_DATA)).setModelName("GLOVE").build();
        modelTrainer.trainModel(modelParameters);
        word2Vec.saveModel(INPUT_MODEL);
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
