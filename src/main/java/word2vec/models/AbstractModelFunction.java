package word2vec.models;

import com.expleague.commons.math.FuncC1.Stub;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import word2vec.text_utils.Cooccurences;
import word2vec.text_utils.Vocabulary;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Array;

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

    public abstract void trainModel();

    public abstract void saveModel(String filepath) throws IOException;

    public abstract void loadModel(String filepath) throws IOException;

    public abstract ArrayVec getVectorByWord(String word);

    public abstract String getWordByVector(ArrayVec vector);

    ArrayVec sumVecs(ArrayVec v1, ArrayVec v2) {
        if (v1.dim() != v2.dim()) {
            throw new RuntimeException("Trying to sum vectors with different dimensions");
        }
        int n = v1.dim();
        ArrayVec res = new ArrayVec(n);
        for (int i = 0; i < n; i++) {
            res.set(i, v1.get(i) + v2.get(i));
        }
        return res;
    }

    double countVecNorm(ArrayVec vec) {
        double res = 0d;
        for (int j = 0; j < vocab_size; j++) {
            final double dv = vec.get(j);
            res += dv * dv;
        }
        return res;
    }

    /*ArrayVec[] squareVecMatrices(ArrayVec[] mat) {
        int n = mat.length;
        int m = mat[0].dim();
        ArrayVec[] res = new ArrayVec[n];
        for (int i = 0; i < n; i ++) {
            double[] row = new double[n];
            for (int j = 0; j < n; j++) {
                double a = 0d;
                for (int k = 0; k <= i; k++) {
                    a += mat[k].get(i) * mat[k].get(j);
                }
                row[j] = a;
            }
            res[i] = new ArrayVec(row);
        }
        return res;
    }*/

    void writeArrayVec(ArrayVec vec, PrintStream fout) throws IOException {
        StringBuilder str = new StringBuilder();
        for (int j = 0; j < vec.dim(); j++) {
            str.append(vec.get(j));
            str.append("\t");
        }
        fout.println(str.toString());
    }

    ArrayVec readArrayVec(BufferedReader fin) throws IOException {
        String[] values = fin.readLine().split("\t");
        ArrayVec vec = new ArrayVec(values.length);
        for (int j = 0; j < values.length; j++)
            vec.set(j, Double.parseDouble(values[j]));
        return vec;
    }
}
