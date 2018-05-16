import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import word2vec.Word2Vec;

import java.io.IOException;

public class Main {

    private static String smallText = "data/corpuses/small_text.txt";
    private static String stardust = "data/corpuses/stardust.txt";
    private static String hobbit = "data/corpuses/hobbit.txt";
    private static String wikiforia_dump = "data/corpuses/wikiforia_dump.txt";

    public static void main(String[] args) throws IOException {
        Word2Vec word2Vec = new Word2Vec();
        /*Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.buildVocab(hobbit);
        System.out.println(word2Vec.vocabSize());
        System.out.println(word2Vec.vocab());
        modelTrainer.trainModel(hobbit, "DECOMP");
        word2Vec.saveModel("data/models/hobbit");
        /*word2Vec.loadModel("data/models/hobbit");
        System.out.println(word2Vec.vocabSize());
        System.out.println(word2Vec.vocab());
        Word2Vec.Model model = word2Vec.getModel();*/
        word2Vec.loadModel("data/models/hobbit");
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.trainModel(hobbit, "DECOMP");
        word2Vec.saveModel("data/models/hobbit");
    }
}
