import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import word2vec.Word2Vec;
import word2vec.exceptions.Word2VecUsageException;

import java.io.IOException;
import java.io.*;

public class Main {

    private static String smallText = "data/corpuses/small_text.txt";
    private static String stardust = "data/corpuses/stardust.txt";
    private static String hobbit = "data/corpuses/hobbit.txt";
    private static String wikiforia_dump = "data/corpuses/wikiforia_dump.txt";

    public static void main(String[] args) throws IOException {
        Word2Vec word2Vec = new Word2Vec();
        /*Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.buildVocab(wikiforia_dump);
        System.out.println(word2Vec.vocabSize());
        modelTrainer.trainModel(wikiforia_dump, "DECOMP");
        word2Vec.saveModel("data/models/wikiforia");
        /*word2Vec.loadModel("data/models/hobbit");
        System.out.println(word2Vec.vocabSize());
        System.out.println(word2Vec.vocab());
        Word2Vec.Model model = word2Vec.getModel();*/
        word2Vec.loadModel("data/models/hobbit");
        Word2Vec.Model model = word2Vec.getModel();
        System.out.println(model.getClosest("friend"));
        System.out.println(model.addWord("elf", "friend"));
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
        }
        /*word2Vec.loadModel("data/models/wikiforia");
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.trainModel(wikiforia_dump, "DECOMP");
        word2Vec.saveModel("data/models/wikiforia");*/
    }
}
