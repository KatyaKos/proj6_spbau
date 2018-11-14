package word2vec.models;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import word2vec.exceptions.LoadingModelException;
import word2vec.text_utils.ArrayVector;
import word2vec.text_utils.Vocabulary;

import java.io.*;
import java.util.List;
import java.util.stream.IntStream;

public class GloveModelFunction extends AbstractModelFunction {
    final private static int TRAINING_ITERS = 20;
    final private static double TRAINING_STEP_COEFF = -0.000001;

    private final static int VECTOR_SIZE = 25;
    private final static double WEIGHTING_X_MAX = 100;
    private final static double WEIGHTING_ALPHA = 0.75;

    private Mx leftVectors;
    private Mx rightVectors;


    public GloveModelFunction(Vocabulary voc, Mx coocc) {
        this(voc, coocc, VECTOR_SIZE);
    }

    public GloveModelFunction(Vocabulary voc, Mx coocc, int vectorSize) {
        super(voc, coocc);
        leftVectors = new VecBasedMx(voc.size(), vectorSize);
        rightVectors = new VecBasedMx(voc.size(), vectorSize);
        for (int i = 0; i < voc.size(); i++) {
            for (int j = 0; j < vectorSize; j++) {
                leftVectors.set(i, j, 1d + Math.random());
                rightVectors.set(i, j, 1d + Math.random());
            }
        }
    }

    @Override
    public Vec getVectorByWord(String word) {
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
        return IntStream.range(0, crcLeft.rows()).parallel().mapToDouble(i -> {
            final VecIterator nz = crcLeft.row(i).nonZeroes();
            double res = 0;
            while (nz.advance()) {
                final int j = nz.index();
                final double X_ij = nz.value();
                res += weightingFunc(X_ij) * (VecTools.multiply(leftVectors.row(i), rightVectors.row(j)) - Math.log(1d + X_ij));
            }
            return res;
        }).sum();
    }

    @Override
    public void saveModel(String filepath) throws IOException {
        File file = new File(filepath);
        PrintStream fout = new PrintStream(file);
        fout.println("GLOVE");
        fout.println("!!! LEFT !!!");
        for (int i = 0; i < vocab_size; i++)
            ArrayVector.writeArrayVec(leftVectors.row(i), fout);
        fout.println("!!! RIGHT !!!");
        for (int i = 0; i < vocab_size; i++)
            ArrayVector.writeArrayVec(rightVectors.row(i), fout);
        fout.close();
    }

    @Override
    public void loadModel(String filepath) throws IOException {
        try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath)))){
            fin.readLine();
            fin.readLine();
            //noinspection Duplicates
            for (int i = 0; i < vocab_size; i++) {
                final Vec vec = ArrayVector.readArrayVec(fin);
                if (leftVectors == null)
                    leftVectors = new VecBasedMx(vocab_size, vec.dim());
                VecTools.assign(leftVectors.row(0), vec);
            }

            fin.readLine();
            //noinspection Duplicates
            for (int i = 0; i < vocab_size; i++) {
                final Vec vec = ArrayVector.readArrayVec(fin);
                if (rightVectors == null)
                    rightVectors = new VecBasedMx(vocab_size, vec.dim());
                VecTools.assign(rightVectors.row(0), vec);
            }
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find vocabulary file to load the model from.");
        }
    }

    private double weightingFunc(double x) {
        return Math.pow((x / WEIGHTING_X_MAX), WEIGHTING_ALPHA);
    }

    @Override
    public void trainModel() {
        Mx dLeft = new VecBasedMx(leftVectors.rows(), leftVectors.columns());
        Mx dRight = new VecBasedMx(rightVectors.rows(), rightVectors.columns());
        for (int iter = 0; iter < TRAINING_ITERS; iter++) {
            VecTools.fill(dLeft, 0);
            VecTools.fill(dRight, 0);

            IntStream.range(0, crcLeft.rows()).parallel().forEach(i -> { // left part derivative
                final VecIterator nz = crcLeft.row(i).nonZeroes();
                final Vec dLeft_i = dLeft.row(i);
                while (nz.advance()) {
                    int j = nz.index();
                    double asum = VecTools.multiply(leftVectors.row(i), rightVectors.row(j));
                    final double X_ij = nz.value();
                    VecTools.incscale(dLeft_i, rightVectors.row(j), weightingFunc(X_ij) * (asum - Math.log(1d + X_ij)));
                }
            });
            IntStream.range(0, crcRight.rows()).parallel().forEach(j -> { // right part derivative
                final VecIterator nz = crcRight.row(j).nonZeroes();
                final Vec dRight_j = dRight.row(j);
                while (nz.advance()) {
                    int i = nz.index();
                    if (i == j)
                        continue;
                    double asum = VecTools.multiply(leftVectors.row(i), rightVectors.row(j));
                    final double X_ij = nz.value();
                    VecTools.incscale(dRight_j, leftVectors.row(i), weightingFunc(X_ij) * (asum - Math.log(1d + X_ij)));
                }
            });

            VecTools.incscale(leftVectors, dLeft, TRAINING_STEP_COEFF);
            VecTools.incscale(rightVectors, dRight, TRAINING_STEP_COEFF);
            System.out.println("Gradient norm: " + Math.sqrt(VecTools.sum2(dLeft) + VecTools.sum2(dRight)));
        }
        //System.out.println("Likelihood: " + likelihood());
    }

    @Override
    public double value(Vec vec) {
        return likelihood();
    }

    @Override
    public int dim() {
        return vocab_size;
    }
}
