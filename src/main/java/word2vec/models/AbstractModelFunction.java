package word2vec.models;

import com.expleague.commons.math.FuncC1.Stub;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import word2vec.exceptions.LoadingModelException;
import word2vec.text_utils.ArrayVector;
import word2vec.text_utils.Cooccurences;
import word2vec.text_utils.Vocabulary;

import java.io.*;
import java.lang.reflect.Array;
import java.util.List;

// Model J = sum[ f(Xij) * (viT*uj - logXij)^2]
//TODO stochastic gradient
public abstract class AbstractModelFunction extends Stub {

    final Vocabulary vocab;
    final Cooccurences crcs;
    final int vocab_size;
    final int vector_size;

    public AbstractModelFunction(Vocabulary vocab, Cooccurences crcs, int vector_size) {
        this.vocab = vocab;
        this.crcs = crcs;
        this.vocab_size = vocab.size();
        this.vector_size = vector_size;
    }

    public abstract void prepareReadyModel();

    public abstract void trainModel();

    public abstract void saveModel(String filepath) throws IOException;

    public abstract void loadModel(String filepath) throws IOException;

    public abstract ArrayVec getVectorByWord(String word);

    public abstract List<String> getWordByVector(ArrayVec vector);

    public abstract double likelihood();

    public abstract double getDistance(String from, String to);

    public abstract double getSkewVector(String word);

    void loadModel(String filepath, ArrayVec[] arr1, ArrayVec[] arr2) throws IOException {
        File file = new File(filepath);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find vocabulary file to load the model from.");
        }
        fin.readLine();
        fin.readLine();
        for (int i = 0; i < vocab_size; i++)
            arr1[i] = ArrayVector.readArrayVec(fin);
        fin.readLine();
        for (int i = 0; i < vocab_size; i++)
            arr2[i] = ArrayVector.readArrayVec(fin);
        fin.close();
    }
}
