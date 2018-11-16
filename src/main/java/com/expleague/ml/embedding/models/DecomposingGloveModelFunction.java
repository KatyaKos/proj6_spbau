package com.expleague.ml.embedding.models;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.logging.Interval;
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
  final private static int TRAINING_ITERS = 300;
  private static final double LAMBDA = 0.0001;
  private static double TRAINING_STEP_COEFF = 1e-2;

  private final static double WEIGHTING_X_MAX = 100;
  private final static double WEIGHTING_ALPHA = 0.75;

  private Mx symDecomp;
  private Mx skewsymDecomp;


  public DecomposingGloveModelFunction(Vocabulary voc, Mx coocc) {
    this(voc, coocc, 40, 10);
  }

  public DecomposingGloveModelFunction(Vocabulary voc, Mx coocc, int symDim, int asymDim) {
    super(voc, coocc);
    symDecomp = new VecBasedMx(vocab_size, symDim);
    skewsymDecomp = new VecBasedMx(vocab_size, asymDim);
    for (int i = 0; i < vocab_size; i++) {
      for (int j = 0; j < symDecomp.columns(); j++) {
        symDecomp.set(i, j, Math.random());
      }
      for (int j = 0; j < skewsymDecomp.columns(); j++) {
        skewsymDecomp.set(i, j, Math.random());
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
    try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath)))) {
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
    }
    catch (FileNotFoundException e) {
      throw new LoadingModelException("Couldn't find vocabulary file to load the model from.");
    }
  }

  private double weightingFunc(double x) {
    return x < WEIGHTING_X_MAX ? Math.pow(x / WEIGHTING_X_MAX, WEIGHTING_ALPHA) : 1;
  }

  @Override
  public void trainModel() {
    Mx softMaxSym = new VecBasedMx(symDecomp.rows(), symDecomp.columns());
    Mx softMaxSkewsym = new VecBasedMx(skewsymDecomp.rows(), skewsymDecomp.columns());
    VecTools.fill(softMaxSym, 1);
    VecTools.fill(softMaxSkewsym, 1);
    for (int iter = 0; iter < TRAINING_ITERS; iter++) {
      Interval.start();
      final double[] counter = new double[]{0, 0};
      double score = IntStream.range(0, crcLeft.rows()).parallel().mapToDouble(i -> { // left part derivative
        final VecIterator nz = crcLeft.row(i).nonZeroes();
        final Vec sym_i = symDecomp.row(i);
        final Vec skew_i = skewsymDecomp.row(i);
        final Vec softMaxSym_i = softMaxSym.row(i);
        final Vec softMaxSkew_i = softMaxSkewsym.row(i);

        double totalScore = 0;
        double totalWeight = 0;
        double totalCount = 0;
        while (nz.advance()) {
          int j = nz.index();
          final Vec sym_j = symDecomp.row(j);
          final Vec skew_j = skewsymDecomp.row(j);
          double asum = VecTools.multiply(sym_i, sym_j);
          double bsum = VecTools.multiply(skew_i, skew_j);
          final double X_ij = nz.value();
          final int sign = i > j ? -1 : 1;
          final double diff = asum + sign * bsum - Math.log(1d + X_ij);
          final double weight = weightingFunc(X_ij);
          totalWeight += weight;
          totalCount ++;
          totalScore += weight * diff * diff;
          IntStream.range(0, sym_i.dim()).forEach(id -> {
            final double d = TRAINING_STEP_COEFF * weight * diff * sym_j.get(id);
            sym_i.adjust(id, -d / Math.sqrt(softMaxSym_i.get(id)));
            softMaxSym_i.adjust(id, d * d);
          });
          IntStream.range(0, skew_i.dim()).forEach(id -> {
            final double d = TRAINING_STEP_COEFF * sign * weight * diff * skew_j.get(id);
            final double x_t = skew_i.get(id) - d / Math.sqrt(softMaxSkew_i.get(id));
            skew_i.set(id, project(x_t));
            softMaxSkew_i.adjust(id, d * d);
          });
        }
        synchronized (counter) {
          counter[0] += totalWeight;
          counter[1] += totalCount;
        }
        return totalScore;
      }).sum();
      IntStream.range(0, crcRight.rows()).parallel().forEach(j -> { // right part derivative
        final Vec sym_j = symDecomp.row(j);
        final Vec skew_j = skewsymDecomp.row(j);
        final Vec softMaxSym_j = softMaxSym.row(j);
        final Vec softMaxSkew_j = softMaxSkewsym.row(j);

        final VecIterator nz = crcRight.row(j).nonZeroes();
        while (nz.advance()) {
          int i = nz.index();
          if (i == j)
            continue;
          final Vec sym_i = symDecomp.row(i);
          final Vec skew_i = skewsymDecomp.row(i);
          double asum = VecTools.multiply(symDecomp.row(i), symDecomp.row(j));
          double bsum = VecTools.multiply(skewsymDecomp.row(i), skewsymDecomp.row(j));
          final double X_ij = nz.value();
          final int sign = i > j ? -1 : 1;
          final double weight = weightingFunc(X_ij);
          final double diff = asum + sign * bsum - Math.log(1d + X_ij);

          IntStream.range(0, sym_j.dim()).forEach(id -> {
            final double d = TRAINING_STEP_COEFF * weight * diff * sym_i.get(id);
            sym_j.adjust(id, -d / Math.sqrt(softMaxSym_j.get(id)));
            softMaxSym_j.adjust(id, d * d);
          });
          IntStream.range(0, skew_j.dim()).forEach(id -> {
            final double d = TRAINING_STEP_COEFF * sign * weight * diff * skew_i.get(id);
            final double x_t = skew_j.get(id) - d / Math.sqrt(softMaxSkew_j.get(id));
            skew_j.set(id, project(x_t));
            softMaxSkew_j.adjust(id, d * d);
          });
        }
      });

      Interval.stopAndPrint("Score: " + (score / counter[1]));
    }
    //System.out.println("Likelihood: " + likelihood());
  }

  private double project(double x_t) {
    return x_t > LAMBDA ? x_t - LAMBDA : (x_t < -LAMBDA ? x_t + LAMBDA : 0);
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