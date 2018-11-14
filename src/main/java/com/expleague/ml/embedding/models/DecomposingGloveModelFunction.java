package com.expleague.ml.embedding.models;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.embedding.exceptions.LoadingModelException;
import com.expleague.ml.embedding.exceptions.Word2VecUsageException;
import com.expleague.ml.embedding.text_utils.ArrayVector;
import com.expleague.ml.embedding.text_utils.Vocabulary;

import java.io.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.expleague.ml.embedding.text_utils.ArrayVector.writeArrayVec;

public class DecomposingGloveModelFunction extends AbstractModelFunction {

    final private static int TRAINING_ITERS = 10;
    private static double TRAINING_STEP_COEFF = -0.0001;

    private final static int VECTOR_SIZE = 25;
    private final static double WEIGHTING_X_MAX = 100;
    private final static double WEIGHTING_ALPHA = 0.75;

    private Mx symDecomp;
    private Mx skewsymDecomp;


    public DecomposingGloveModelFunction(Vocabulary voc, Mx coocc) {
        this(voc, coocc, 50, 10);
    }

    public DecomposingGloveModelFunction(Vocabulary voc, Mx coocc, int symDim, int asymDim) {
        super(voc, coocc);
        symDecomp = new VecBasedMx(vocab_size, symDim);
        skewsymDecomp = new VecBasedMx(vocab_size, asymDim);
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < VECTOR_SIZE; j++) {
                symDecomp.set(i, j, 1d + Math.random());
                skewsymDecomp.set(i, j, 1d + Math.random());
            }
        }
    }

    @Override
    public Vec getVectorByWord(String word) {
        int w = vocab.wordToIndex(word);
        if (w == Vocabulary.NO_ENTRY_VALUE)
            throw new Word2VecUsageException("There's no word " + word + " in the vocabulary.");
        return symDecomp.row(w);
    }

    @Override
    public List<String> getWordByVector(Vec vector) {
        int[] order = ArrayTools.sequence(0, vocab_size);
        double[] weights = IntStream.of(order).mapToDouble(idx -> -VecTools.cosine(symDecomp.row(idx), vector)).toArray();
        ArrayTools.parallelSort(weights, order);
        return IntStream.range(0, 5).mapToObj(idx -> vocab.indexToWord(order[idx])).collect(Collectors.toList());
        /*int[] order = ArrayTools.sequence(0, vocab_size);
        double[] weights = IntStream.of(order).mapToDouble(idx ->
                countVecNorm(vectorsDifference(symDecomp[idx], vector))).toArray();
        ArrayTools.parallelSort(weights, order);
        return IntStream.range(0, 5).mapToObj(idx -> vocab.indexToWord(order[idx])).collect(Collectors.toList());*/
    }

    @Override
    public double getSkewVector(String word) {
        int w = vocab.wordToIndex(word);
//        double result = 0;
//        for (int i = 0; i < vocab_size; i++) {
//            if (i == w)
//                continue;
//            result += VecTools.multiply(skewsymDecomp[i], skewsymDecomp[w]);
//        }
        return VecTools.norm(skewsymDecomp.row(w)); //result;
    }

    @Override
    public double getDistance(String from, String to) {
        return VecTools.cosine(getVectorByWord(from), getVectorByWord(to));
    }

    @Override
    public double likelihood() {
        double res = 0d;
        for (int i = 0; i < vocab_size; i++) {
            final VecIterator nz = crcLeft.row(i).nonZeroes();
            final Vec u_i = symDecomp.row(i);
            final Vec v_i = skewsymDecomp.row(i);
            while (nz.advance()) {
                double xij = nz.value();
                int j = nz.index();
                double v = VecTools.multiply(u_i, symDecomp.row(j));
                double u = VecTools.multiply(v_i, skewsymDecomp.row(j));
                if (i > j) u *= -1d;
                double diff = v + u - Math.log(1d + xij);
                res += weightingFunc(xij) * diff * diff;
            }
        }
        return res;
    }

    @Override
    public void prepareReadyModel() {
    }

    @Override
    public void saveModel(String filepath) throws IOException {
        File file = new File(filepath);
        PrintStream fout = new PrintStream(file);
        fout.println("DECOMP");
        fout.println("!!! SYMMETRIC !!!");
        for (int i = 0; i < vocab_size; i++)
            writeArrayVec(symDecomp.row(i), fout);
        fout.println("!!! SKEWSYMMETRIC !!!");
        for (int i = 0; i < vocab_size; i++)
            writeArrayVec(skewsymDecomp.row(i), fout);
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
                if (symDecomp == null)
                    symDecomp = new VecBasedMx(vocab_size, vec.dim());
                VecTools.assign(symDecomp.row(0), vec);
            }

            fin.readLine();
            //noinspection Duplicates
            for (int i = 0; i < vocab_size; i++) {
                final Vec vec = ArrayVector.readArrayVec(fin);
                if (skewsymDecomp == null)
                    skewsymDecomp = new VecBasedMx(vocab_size, vec.dim());
                VecTools.assign(skewsymDecomp.row(0), vec);
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
        Mx dSym = new VecBasedMx(symDecomp.rows(), symDecomp.columns());
        Mx dSkewsym = new VecBasedMx(skewsymDecomp.rows(), skewsymDecomp.columns());
        for (int iter = 0; iter < TRAINING_ITERS; iter++) {
            VecTools.fill(dSym, 0);
            VecTools.fill(dSkewsym, 0);

            long[] counter = new long[]{0};
            IntStream.range(0, crcLeft.rows()).parallel().forEach(i -> { // left part derivative
                final VecIterator nz = crcLeft.row(i).nonZeroes();
                final Vec dSym_i = dSym.row(i);
                final Vec dSkew_i = dSkewsym.row(i);
                while (nz.advance()) {
                    counter[0]++;
                    int j = nz.index();
                    double asum = VecTools.multiply(symDecomp.row(i), symDecomp.row(j));
                    double bsum = VecTools.multiply(skewsymDecomp.row(i), skewsymDecomp.row(j));
                    final double X_ij = nz.value();
                    final int sign = i > j ? -1 : 1;
                    VecTools.incscale(dSym_i, symDecomp.row(j), weightingFunc(X_ij) * (asum + sign * bsum - Math.log(1d + X_ij)));
                    VecTools.incscale(dSkew_i, skewsymDecomp.row(j), sign * weightingFunc(X_ij) * (asum + sign * bsum - Math.log(1d + X_ij)));
                }
            });
            IntStream.range(0, crcRight.rows()).parallel().forEach(j -> { // right part derivative
                final VecIterator nz = crcRight.row(j).nonZeroes();
                final Vec dSym_j = dSym.row(j);
                final Vec dSkew_j = dSkewsym.row(j);
                while (nz.advance()) {
                    int i = nz.index();
                    if (i == j)
                        continue;
                    double asum = VecTools.multiply(symDecomp.row(i), symDecomp.row(j));
                    double bsum = VecTools.multiply(skewsymDecomp.row(i), skewsymDecomp.row(j));
                    final double X_ij = nz.value();
                    final int sign = i > j ? -1 : 1;
                    VecTools.incscale(dSym_j, symDecomp.row(i), weightingFunc(X_ij) * (asum + sign * bsum - Math.log(1d + X_ij)));
                    VecTools.incscale(dSkew_j, skewsymDecomp.row(i), sign * weightingFunc(X_ij) * (asum + sign * bsum - Math.log(1d + X_ij)));
                }
            });

            VecTools.incscale(symDecomp, dSym, TRAINING_STEP_COEFF);
            VecTools.incscale(skewsymDecomp, dSkewsym, TRAINING_STEP_COEFF);
            System.out.println("Gradient norm: " + Math.sqrt((VecTools.sum2(dSym) + VecTools.sum2(dSkewsym)) / counter[0]));
        }
        //System.out.println("Likelihood: " + likelihood());
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
