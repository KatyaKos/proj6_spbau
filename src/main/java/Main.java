import word2vec.Word2Vec;

import java.io.IOException;

public class Main {

    private static String smallText = "data/corpuses/hobbit.txt";

    public static void main(String[] args) throws IOException {
        Word2Vec word2Vec = new Word2Vec();
        /*modelTrainer.buildVocab(smallText);
        System.out.println(word2Vec.vocabSize());
        System.out.println(word2Vec.vocab());
        modelTrainer.trainModel(smallText, "DECOMP");
        word2Vec.saveModel("data/models/hobbit");
        word2Vec.loadModel("data/models");
        System.out.println(word2Vec.vocabSize());
        System.out.println(word2Vec.vocab());
        Word2Vec.Model model = word2Vec.getModel();
        System.out.println(model.vecMinusVec("star", "sky"));*/
        word2Vec.loadModel("data/models/hobbit");
        Word2Vec.ModelTrainer modelTrainer = word2Vec.createTrainer();
        modelTrainer.trainModel(smallText, "DECOMP");
        word2Vec.saveModel("data/models/hobbit");


    }
}
