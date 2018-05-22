package word2vec.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import word2vec.exceptions.LoadingModelException;
import word2vec.exceptions.Word2VecUsageException;
import word2vec.text_utils.Cooccurences;
import word2vec.text_utils.Vocabulary;
import word2vec.text_utils.ArrayVector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DecomposingGloveModelFunction extends AbstractModelFunction{

    final private static int TRAINING_ITERS = 10;
    private static double TRAINING_STEP_COEFF = -0.01;
    private static boolean IS_STOCHASTIC = false;

    private final static int VECTOR_SIZE = 25;
    private final static double WEIGHTING_X_MAX = 100;
    private final static double WEIGHTING_ALPHA = 0.75;

    private ArrayVec[] symDecomp;
    private ArrayVec[] skewsymDecomp;
    private ArrayVec[] wordsRepresentation;


    public DecomposingGloveModelFunction(Vocabulary voc, Cooccurences coocc) {
        super(voc, coocc, VECTOR_SIZE);
        symDecomp = new ArrayVec[vocab_size];
        skewsymDecomp = new ArrayVec[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            symDecomp[i] = new ArrayVec(VECTOR_SIZE);
            skewsymDecomp[i] = new ArrayVec(VECTOR_SIZE);
            for (int j = 0; j < VECTOR_SIZE; j++) {
                symDecomp[i].set(j, 1d + Math.random());
                skewsymDecomp[i].set(j, 1d + Math.random());
            }
        }
    }

    @Override
    public ArrayVec getVectorByWord(String word) {
        int w = vocab.wordToIndex(word);
        if (w == Vocabulary.NO_ENTRY_VALUE)
            throw new Word2VecUsageException("There's no word " + word + " in the vocabulary.");
        return symDecomp[w];
    }

    @Override
    public String getWordByVector(ArrayVec vector) {
        double minNorm = Double.MAX_VALUE;
        int closest = -1;
        for (int i = 0; i < vocab_size; i++) {
            final ArrayVec vec = symDecomp[i];
            final double norm = ArrayVector.countVecNorm(ArrayVector.vectorsDifference(vector, vec));
            if (norm < minNorm) {
                minNorm = norm;
                closest = i;
            }
        }
        if (closest == -1) {
            return null;
        }
        return vocab.indexToWord(closest);
    }

    @Override
    public void prepareReadyModel() {
        wordsRepresentation = new ArrayVec[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            wordsRepresentation[i] = ArrayVector.sumVectors(symDecomp[i], skewsymDecomp[i]);
        }
    }

    @Override
    public void saveModel(String filepath) throws IOException {
        File file = new File(filepath);
        PrintStream fout = new PrintStream(file);
        fout.println("DECOMP");
        fout.println("!!! SYMMETRIC !!!");
        for (int i = 0; i < vocab_size; i++)
            ArrayVector.writeArrayVec(symDecomp[i], fout);
        fout.println("!!! SKEWSYMMETRIC !!!");
        for (int i = 0; i < vocab_size; i++)
            ArrayVector.writeArrayVec(skewsymDecomp[i], fout);
        fout.close();
    }

    @Override
    public void loadModel(String filepath) throws IOException {
        loadModel(filepath, symDecomp, skewsymDecomp);
    }

    private double weightingFunc(double x) {
        return Math.pow((x / WEIGHTING_X_MAX), WEIGHTING_ALPHA);
    }

    @Override
    public void trainModel() {
        Random random = new Random();
        double norm2 = Double.MAX_VALUE;
        if (IS_STOCHASTIC) {
            for (int iter = 0; iter < TRAINING_ITERS; iter++) {
                double[] dSym = new double[vocab_size];
                double[] dSkewsym = new double[vocab_size];
                double norm = 0d;
                int derivativeIndex = random.nextInt(vector_size);
                System.out.println(derivativeIndex);
                for (int i = 0; i < vocab_size; i++) {
                    dSym[i] = countDerivative(i, derivativeIndex, true);
                    norm += vector_size * dSym[i] * dSym[i];
                    dSym[i] *= TRAINING_STEP_COEFF;
                    dSkewsym[i] = countDerivative(i, derivativeIndex, false);
                    norm += vector_size * dSkewsym[i] * dSkewsym[i];
                    dSkewsym[i] *= TRAINING_STEP_COEFF;
                }
                if (norm == Double.POSITIVE_INFINITY || Double.isNaN(norm) || norm > norm2) {
                    break;
                }
                norm2 = norm;
                for (int i = 0; i < vocab_size; i++) {
                    for (int j = 0; j < vector_size; j++) {
                        symDecomp[i].adjust(j, dSym[i]);
                        skewsymDecomp[i].adjust(j, dSkewsym[i]);
                    }
                }
                System.out.println("Gradient norm: " + Math.sqrt(norm));
            }
        } else {
            for (int iter = 0; iter < TRAINING_ITERS; iter++) {
                ArrayVec[] dSym = new ArrayVec[vocab_size];
                ArrayVec[] dSkewsym = new ArrayVec[vocab_size];
                double norm = 0d;
                for (int i = 0; i < vocab_size; i++) {
                    dSym[i] = countDerivative(i, true);
                    norm += ArrayVector.countVecNorm(dSym[i]);
                    dSym[i].scale(TRAINING_STEP_COEFF);
                    dSkewsym[i] = countDerivative(i, false);
                    norm += ArrayVector.countVecNorm(dSkewsym[i]);
                    dSkewsym[i].scale(TRAINING_STEP_COEFF);
                }
                if (norm == Double.POSITIVE_INFINITY || Double.isNaN(norm) || norm > norm2) {
                    break;
                }
                norm2 = norm;
                for (int i = 0; i < vocab_size; i++) {
                    symDecomp[i] = ArrayVector.sumVectors(symDecomp[i], dSym[i]);
                    skewsymDecomp[i] = ArrayVector.sumVectors(skewsymDecomp[i], dSkewsym[i]);
                }
                System.out.println("Gradient norm: " + Math.sqrt(norm));
            }
        }
        //System.out.println("Likelihood: " + likelihood());
    }

    private double likelihood() {
        double res = 0d;
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < vocab_size; j++) {
                double diff;
                double v = symDecomp[i].mul(symDecomp[j]);
                double u = skewsymDecomp[i].mul(skewsymDecomp[j]);
                if (i > j) u *= -1d;
                double xij = crcs.getValue(i, j);
                diff = v + u - Math.log(1d + xij);
                res += weightingFunc(xij) * diff * diff;
            }
        }
        return res;
    }

    private ArrayVec countDerivative(int i, boolean isSym) {
        ArrayVec res = new ArrayVec(vector_size);
        for (int k = i; k < vector_size; k++) {
            res.set(k, countDerivative(i, k, isSym));
        }
        return res;
    }

    //c = vocab_size, r = vector_size
    private double countDerivative(int c, int r, boolean isSym) {
        ArrayVec[] mat;
        if (isSym) mat = symDecomp;
        else mat = skewsymDecomp;
        double x = crcs.getValue(c, c);
        double res = weightingFunc(x) * mat[c].get(r) * 2 *
                (multMatrixLines(c, c, true) + multMatrixLines(c, c, false) - Math.log(1d + x));
        for (int i = 0; i < vocab_size; i++) {
            if (i == c) continue;
            double asum = multMatrixLines(i, c, true);
            double bsum = multMatrixLines(i, c, false);
            double sign = 1d;
            if (!isSym && i > c) sign = -1d;
            x = crcs.getValue(i, c);
            res += weightingFunc(x) * sign * mat[i].get(r) * (asum + sign * bsum - Math.log(1d + x));
            x = crcs.getValue(c, i);
            if (!isSym && i < c) sign = -1d;
            else sign = 1d;
            res += weightingFunc(x) * sign * mat[i].get(r) * (asum + sign * bsum - Math.log(1d + x));
        }
        return res;
    }

    private double multMatrixLines(int i, int j, boolean isSym) {
        ArrayVec[] mat;
        if (isSym) mat = symDecomp;
        else mat = skewsymDecomp;
        double sum = 0d;
        for (int l = 0; l < vector_size; l++) {
            sum += mat[j].get(l) * mat[i].get(l);
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
