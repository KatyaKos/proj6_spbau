package word2vec.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import word2vec.exceptions.LoadingModelException;
import word2vec.text_utils.ArrayVector;
import word2vec.text_utils.Cooccurences;
import word2vec.text_utils.Vocabulary;

import java.io.*;

public class GloveModelFunction extends AbstractModelFunction {

    final private static int TRAINING_ITERS = 20;
    final private static double TRAINING_STEP_COEFF = -0.01;

    private final static int VECTOR_SIZE = 10;
    private final static double START_VECTOR_VALUE = 2d;
    private final static double WEIGHTING_X_MAX = 100;
    private final static double WEIGHTING_ALPHA = 0.75;

    private ArrayVec[] leftVectors;
    private ArrayVec[] rightVectors;


    public GloveModelFunction(Vocabulary voc, Cooccurences coocc) {
        super(voc, coocc, VECTOR_SIZE);
        leftVectors = new ArrayVec[vocab_size];
        rightVectors = new ArrayVec[vocab_size];
        double[] tmp = new double[VECTOR_SIZE];
        for (int i = 0; i < VECTOR_SIZE; i++) tmp[i] = START_VECTOR_VALUE;
        for (int i = 0; i < vocab_size; i++) {
            leftVectors[i] = new ArrayVec(tmp);
            rightVectors[i] = new ArrayVec(tmp);
        }
    }

    @Override
    public ArrayVec getVectorByWord(String word) {
        return null;
    }

    @Override
    public String getWordByVector(ArrayVec vector) {
        return null;
    }

    @Override
    public void prepareReadyModel() {

    }

    @Override
    public void saveModel(String filepath) throws IOException {
        File file = new File(filepath);
        PrintStream fout = new PrintStream(file);
        fout.println("GLOVE");
        fout.println("!!! LEFT !!!");
        for (int i = 0; i < vocab_size; i++)
            ArrayVector.writeArrayVec(leftVectors[i], fout);
        fout.println("!!! RIGHT !!!");
        for (int i = 0; i < vocab_size; i++)
            ArrayVector.writeArrayVec(rightVectors[i], fout);
        fout.close();
    }

    @Override
    public void loadModel(String filepath) throws IOException {
        loadModel(filepath, leftVectors, rightVectors);
    }

    private double weightingFunc(double x) {
        return Math.pow((x / WEIGHTING_X_MAX), WEIGHTING_ALPHA);
    }

    @Override
    public void trainModel() {
        for (int iter = 0; iter < TRAINING_ITERS; iter++) {
            ArrayVec[] dLeftVecs = new ArrayVec[vocab_size];
            ArrayVec[] dRightVecs = new ArrayVec[vocab_size];
            double norm = 0d;
            for (int i = 0; i < vocab_size; i++) {
                dLeftVecs[i] = countVecDerivative(i, true);
                dRightVecs[i] = countVecDerivative(i, false);
                norm += ArrayVector.countVecNorm(dLeftVecs[i]) + ArrayVector.countVecNorm(dRightVecs[i]);
                dLeftVecs[i].scale(TRAINING_STEP_COEFF);
                leftVectors[i] = ArrayVector.sumVectors(leftVectors[i], dLeftVecs[i]);
                dRightVecs[i].scale(TRAINING_STEP_COEFF);
                rightVectors[i] = ArrayVector.sumVectors(rightVectors[i], dRightVecs[i]);
            }
            System.out.println("Grad norm: " + Math.sqrt(norm));
        }
    }

    private ArrayVec countVecDerivative(int i, boolean isLeft) {
        ArrayVec res = new ArrayVec(vector_size);
        for (int k = 0; k < vector_size; k++) {
            res.set(k, countVecDerivative(i, k, isLeft));
        }
        return res;
    }

    private double countVecDerivative(int i, int k, boolean isLeft) {
        double res = 0d;
        double xij;
        ArrayVec u;
        ArrayVec v;
        if (isLeft) v = leftVectors[i];
        else v = rightVectors[i];
        for (int j = 0; j < vocab_size; j++) {
            if (isLeft) {
                xij = crcs.getValue(i, j);
                u = rightVectors[j];
            } else {
                xij = crcs.getValue(j, i);
                u = leftVectors[j];
            }
            double diff = v.mul(u) - Math.log(xij);
            res += weightingFunc(xij) * 2 * diff * u.get(k);
        }
        return res;
    }

    @Override
    public double value(Vec vec) {
        double res = 0d;
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < vocab_size; j++) {
                double diff = 0d;
                ArrayVec v = leftVectors[i];
                ArrayVec u = rightVectors[j];
                double xij = crcs.getValue(i, j);
                diff = v.mul(u) - Math.log(xij);
                res += weightingFunc(xij) * diff * diff;
            }
        }
        return res;
    }

    @Override
    public int dim() {
        return vocab_size;
    }
}
