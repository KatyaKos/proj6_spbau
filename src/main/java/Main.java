import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import word2vec.Word2Vec;
import word2vec.exceptions.Word2VecUsageException;
import word2vec.quality_metrics.QualityMetric;
import word2vec.quality_metrics.impl.CloserFurtherMetric;

import java.io.IOException;
import java.io.*;

public class Main {

    private static String smallText = "data/corpuses/small_text.txt";
    private static String stardust = "data/corpuses/stardust.txt";
    private static String hobbit = "data/corpuses/hobbit.txt";
    private static String wikiforia_dump = "data/corpuses/wikiforia_dump.txt";

    public static void main(String[] args) throws IOException {
        Word2Vec word2Vec = new Word2Vec();

        // Create new model
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.buildVocab(hobbit);
        System.out.println(word2Vec.vocabSize());
        modelTrainer.trainModel(hobbit, "GLOVE");
        word2Vec.saveModel("data/models/hobbit/glove");

        // Load model and continue training
        /*word2Vec.loadModel("data/models/hobbit/glove");
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.trainModel(hobbit, "GLOVE");
        word2Vec.saveModel("data/models/hobbit/glove");*/

        // Load model and get closest
        /*word2Vec.loadModel("data/models/hobbit");
        Word2Vec.Model model = word2Vec.getModel();
        System.out.println(model.countLikelihood());
        String input;
        LineNumberReader lnr = new LineNumberReader(new InputStreamReader(System.in));
        while ((input = lnr.readLine()) != null) {
            try {
                System.out.println(model.getClosest(input));
            }
            catch (Word2VecUsageException e) {
                System.out.println("No such word");
            }
        }*/

        // Test with closer-further metrics
        /*word2Vec.loadModel("data/models/hobbit");
        Word2Vec.Model model = word2Vec.getModel();
        CloserFurtherMetric metric = new CloserFurtherMetric(model);
        metric.measure("data/tests/hobbit/hobbit_logic.txt","data/tests/hobbit/hobbit_logic_result.txt");*/
    }
}
