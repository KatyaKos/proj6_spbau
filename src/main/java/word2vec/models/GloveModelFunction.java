package word2vec.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.logging.Interval;
import word2vec.exceptions.LoadingModelException;
import word2vec.text_utils.ArrayVector;
import word2vec.text_utils.Cooccurences;
import word2vec.text_utils.Vocabulary;

import java.io.*;
import java.util.List;

public class GloveModelFunction extends AbstractModelFunction {

    final private static int TRAINING_ITERS = 20;
    final private static double TRAINING_STEP_COEFF = -0.000001;

    private final static int VECTOR_SIZE = 25;
    private final static double START_VECTOR_VALUE = 2d;
    private final static double WEIGHTING_X_MAX = 100;
    private final static double WEIGHTING_ALPHA = 0.75;

    private Vec[] leftVectors;
    private Vec[] rightVectors;


    public GloveModelFunction(Vocabulary voc, Cooccurences coocc) {
        super(voc, coocc, VECTOR_SIZE);
        leftVectors = new Vec[vocab_size];
        rightVectors = new Vec[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            leftVectors[i] = new ArrayVec(VECTOR_SIZE);
            rightVectors[i] = new ArrayVec(VECTOR_SIZE);
            for (int j = 0; j < VECTOR_SIZE; j++) {
                leftVectors[i].set(j, 1d + Math.random());
                rightVectors[i].set(j, 1d + Math.random());
            }
        }
    }

    @Override
    public ArrayVec getVectorByWord(String word) {
        return null;
    }

    @Override
    public List<String> getWordByVector(Vec vector) {
        return null;
    }

    @Override
    public void prepareReadyModel() {
    }

    @Override
    public double getDistance(String from, String to) {
        return 0d;
    }

    @Override
    public double getSkewVector(String word) {
        return 0;
    }

    @Override
    public double likelihood() {
        double res = 0d;
        return res;
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
        double norm2 = Double.MAX_VALUE;
        for (int iter = 0; iter < TRAINING_ITERS; iter++) {
            Interval.start();
            Vec[] dLeftVecs = new ArrayVec[vocab_size];
            Vec[] dRightVecs = new ArrayVec[vocab_size];
            double norm = 0d;
            for (int i = 0; i < vocab_size; i++) {
                dLeftVecs[i] = countVecDerivative(i, true);
                dRightVecs[i] = countVecDerivative(i, false);
                norm += VecTools.sum2(dLeftVecs[i]) + VecTools.sum2(dRightVecs[i]);
                VecTools.scale(dLeftVecs[i], TRAINING_STEP_COEFF);
                VecTools.scale(dRightVecs[i], TRAINING_STEP_COEFF);
            }
            if (norm == Double.POSITIVE_INFINITY || Double.isNaN(norm) || norm > norm2) {
                break;
            }
            norm2 = norm;;
            for (int i = 0; i < vocab_size; i++) {
                VecTools.append(leftVectors[i], dLeftVecs[i]);
                VecTools.append(rightVectors[i], dRightVecs[i]);
            }
            Interval.stopAndPrint("Grad norm: " + Math.sqrt(norm));
        }
    }

    private Vec countVecDerivative(int i, boolean isLeft) {
        Vec res = new ArrayVec(vector_size);
        for (int k = 0; k < vector_size; k++) {
            res.set(k, countVecDerivative(i, k, isLeft));
        }
        return res;
    }

    private double countVecDerivative(int i, int k, boolean isLeft) {
        double res = 0d;
        double xij;
        Vec u;
        Vec v;
        if (isLeft) v = leftVectors[i];
        else v = rightVectors[i];
        final int vocab_size = this.vocab_size;
        for (int j = 0; j < vocab_size; j++) {
            if (isLeft) {
                xij = crcs.getValue(i, j);
                u = rightVectors[j];
            } else {
                xij = crcs.getValue(j, i);
                u = leftVectors[j];
            }
            double diff = VecTools.multiply(v, u) - Math.log(1d + xij);
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
                Vec v = leftVectors[i];
                Vec u = rightVectors[j];
                double xij = crcs.getValue(i, j);
                diff = VecTools.multiply(v, u) - Math.log(1d + xij);
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
