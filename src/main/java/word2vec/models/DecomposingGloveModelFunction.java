package word2vec.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import word2vec.exceptions.LoadingModelException;
import word2vec.exceptions.Word2VecUsageException;
import word2vec.text_utils.Cooccurences;
import word2vec.text_utils.Vocabulary;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class DecomposingGloveModelFunction extends AbstractModelFunction{

    final private static int TRAINING_ITERS = 1;
    final private static double TRAINING_STEP_COEFF = -0.01;

    private final static int VECTOR_SIZE = 25;
    private final static double WEIGHTING_X_MAX = 100;
    private final static double WEIGHTING_ALPHA = 0.75;

    private ArrayVec[] symDecomp;
    private ArrayVec[] skewsymDecomp;


    public DecomposingGloveModelFunction(Vocabulary voc, Cooccurences coocc) {
        super(voc, coocc, VECTOR_SIZE);
        symDecomp = new ArrayVec[VECTOR_SIZE];
        skewsymDecomp = new ArrayVec[VECTOR_SIZE];
        for (int i = 0; i < VECTOR_SIZE; i++) {
            symDecomp[i] = new ArrayVec(vocab_size);
            skewsymDecomp[i] = new ArrayVec(vocab_size);
            for (int j = i; j < vocab_size; j++) {
                symDecomp[i].set(j, 2d);
                skewsymDecomp[i].set(j, 2d);
            }
        }
    }

    @Override
    public ArrayVec getVectorByWord(String word) {
        int w = vocab.wordToIndex(word);
        if (w == Vocabulary.NO_ENTRY_VALUE)
            throw new Word2VecUsageException("There's no word " + word + " in the vocabulary.");
        ArrayVec vector = new ArrayVec(vector_size);
        for (int i = 0; i < vector_size; i++)
            vector.set(i, symDecomp[i].get(0));
        return vector;
    }

    @Override
    public String getWordByVector(ArrayVec vector) {
        List<Integer> wordsIndexes = new ArrayList<>(vocab_size);
        for (int i = 0; i < vocab_size; i++) wordsIndexes.add(i);
        for (int i = 0; i < vector_size; i++) {
            List<Integer> new_ind = new ArrayList<>();
            for (Integer j : wordsIndexes)
                if (symDecomp[i].get(j) != vector.get(i))
                    new_ind.add(j);
            wordsIndexes.removeAll(new_ind);
        }
        if (wordsIndexes.isEmpty())
            throw new Word2VecUsageException("No such word found in trained model");
        return vocab.indexToWord(wordsIndexes.get(0));
    }


    @Override
    public void saveModel(String filepath) throws IOException {
        File file = new File(filepath);
        PrintStream fout = new PrintStream(file);
        fout.println("DECOMP");
        fout.println("!!! SYMMETRIC !!!");
        for (int i = 0; i < vector_size; i++)
            writeArrayVec(symDecomp[i], fout);
        fout.println("!!! SKEWSYMMETRIC !!!");
        for (int i = 0; i < vector_size; i++)
            writeArrayVec(skewsymDecomp[i], fout);
        fout.close();
    }

    @Override
    public void loadModel(String filepath) throws IOException {
        File file = new File(filepath);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find vocabulary file to load the model from.");
        }
        fin.readLine();
        fin.readLine();
        symDecomp = new ArrayVec[vector_size];
        skewsymDecomp = new ArrayVec[vector_size];
        for (int i = 0; i < vector_size; i++)
            symDecomp[i] = readArrayVec(fin);
        fin.readLine();
        for (int i = 0; i < vector_size; i++)
            skewsymDecomp[i] = readArrayVec(fin);
        fin.close();
    }

    private double weightingFunc(double x) {
        return Math.pow((x / WEIGHTING_X_MAX), WEIGHTING_ALPHA);
    }

    @Override
    public void trainModel() {
        for (int iter = 0; iter < TRAINING_ITERS; iter++) {
            ArrayVec[] dsymVecs = new ArrayVec[VECTOR_SIZE];
            ArrayVec[] dskewsymVecs = new ArrayVec[VECTOR_SIZE];
            double norm = 0d;
            for (int i = 0; i < VECTOR_SIZE; i++) {
                dsymVecs[i] = countVecDerivative(i, true);
                norm += countVecNorm(dsymVecs[i]);
                dsymVecs[i].scale(TRAINING_STEP_COEFF);
                dskewsymVecs[i] = countVecDerivative(i, false);
                norm += countVecNorm(dskewsymVecs[i]);
                dskewsymVecs[i].scale(TRAINING_STEP_COEFF);
            }
            for (int i = 0; i < VECTOR_SIZE; i++) {
                symDecomp[i] = sumVecs(symDecomp[i], dsymVecs[i]);
                skewsymDecomp[i] = sumVecs(skewsymDecomp[i], dskewsymVecs[i]);
            }
            System.out.println("Gradient norm: " + Math.sqrt(norm));
        }
    }

    private ArrayVec countVecDerivative(int i, boolean isSym) {
        ArrayVec res = new ArrayVec(vocab_size);
        for (int k = i; k < vocab_size; k++) {
            res.set(k, countVecDerivative(i, k, isSym));
        }
        return res;
    }

    private double countVecDerivative(int r, int c, boolean isSym) {
        ArrayVec[] mat;
        if (isSym) mat = symDecomp;
        else mat = skewsymDecomp;
        double x = crcs.getValue(c, c);
        double res = weightingFunc(x) * mat[r].get(c) * 2 *
                (multMatrixLines(c, c, true) + multMatrixLines(c, c, false) - Math.log(1d + x));
        for (int i = 0; i < vocab_size; i++) {
            if (i == c) continue;
            double asum = multMatrixLines(i, c, true);
            double bsum = multMatrixLines(i, c, false);
            double sign = 1d;
            if (!isSym && i > c) sign = -1d;
            x = crcs.getValue(i, c);
            res += weightingFunc(x) * sign * mat[r].get(i) * (asum + sign * bsum - Math.log(1d + x));
            x = crcs.getValue(c, i);
            if (!isSym && i < c) sign = -1d;
            else sign = 1d;
            res += weightingFunc(x) * sign * mat[r].get(i) * (asum + sign * bsum - Math.log(1d + x));
        }
        return res;
    }

    private double multMatrixLines(int i, int j, boolean isSym) {
        ArrayVec[] mat;
        if (isSym) mat = symDecomp;
        else mat = skewsymDecomp;
        double sum = 0d;
        for (int l = 0; l < VECTOR_SIZE; l++) {
            sum += mat[l].get(j) * mat[l].get(i);
        }
        return sum;
    }

    @Override
    public double value(Vec vec) {
        return 0;
    }

    @Override
    public int dim() {
        return vocab_size;
    }
}
