package com.expleague.ml.embedding.model_functions;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.embedding.exceptions.LoadingModelException;
import com.expleague.ml.embedding.text_utils.VecIO;
import com.expleague.ml.embedding.text_utils.Vocabulary;

import java.io.*;
import java.util.stream.IntStream;

public class DecomposingGloveModelFunction extends AbstractModelFunction {
  final private static int TRAINING_ITERS = 100;
  private static final double LAMBDA = 1e-4;
  private static double TRAINING_STEP_COEFF = 1e-1;

  private final static double WEIGHTING_X_MAX = 100;
  private final static double WEIGHTING_ALPHA = 0.75;
  private final static int SYM_DIM = 40;
  private final static int SKEWSYM_DIM = 10;


  private Mx symDecomp = null;
  private Mx skewsymDecomp = null;

  public DecomposingGloveModelFunction(Vocabulary voc, Mx coocc) {
    super(voc, coocc);
  }

  private void initialize() {
    symDecomp = new VecBasedMx(vocab_size, SYM_DIM);
    skewsymDecomp = new VecBasedMx(vocab_size, SKEWSYM_DIM);
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
  public Mx getModelVectors() {
    return symDecomp;
  }

  @Override
  public void saveModel(String filepath) throws IOException {
    File file_sym = new File(filepath + "/sym_vectors.txt");
    File file = new File(filepath + "/vectors.txt");
    PrintStream fout_sym = new PrintStream(file_sym);
    PrintStream fout = new PrintStream(file);
    fout_sym.println("DECOMP");
    for (int i = 0; i < vocab_size; i++) {
      VecIO.writeVec(fout_sym, symDecomp.row(i));
      VecIO.writeVec(fout, symDecomp.row(i));
    }
    for (int i = 0; i < vocab_size; i++) {
      VecIO.writeVec(fout, skewsymDecomp.row(i));
    }
    fout_sym.close();
    fout.close();
  }

  @Override
  public void loadModel(String filepath, int mode) throws IOException {
    if (mode == 0) filepath += "/vectors.txt";
    else if (mode == 1) filepath += "/sym_vectors.txt";

    try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath)))) {
      fin.readLine();
      symDecomp = VecIO.readMx(fin, vocab_size);
      if (mode == 0) {
        skewsymDecomp = VecIO.readMx(fin, vocab_size);
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
    if (symDecomp == null) {
      initialize();
    }
    final Mx softMaxSym = new VecBasedMx(symDecomp.rows(), symDecomp.columns());
    final Mx softMaxSkewsym = new VecBasedMx(skewsymDecomp.rows(), skewsymDecomp.columns());
    final Vec bias = new ArrayVec(crcLeft.rows());
    VecTools.fill(bias, 1);
    final Vec softMaxBias = new ArrayVec(crcLeft.rows());
    for (int iter = 0; iter < TRAINING_ITERS; iter++) {
      VecTools.fill(softMaxSym, 1);
      VecTools.fill(softMaxSkewsym, 1);
      VecTools.fill(softMaxBias, 1);

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
          final Vec softMaxSym_j = softMaxSym.row(j);
          final Vec softMaxSkew_j = softMaxSkewsym.row(j);

          final double b_i = bias.get(i);
          final double b_j = bias.get(j);
          double asum = VecTools.multiply(sym_i, sym_j);
          double bsum = VecTools.multiply(skew_i, skew_j);
          final double X_ij = nz.value();
          final int sign = i > j ? -1 : 1;
          final double minfo = Math.log(X_ij);
          final double diff = MathTools.sqr(b_i) + MathTools.sqr(b_j) + asum + sign * bsum - minfo;
          final double weight = weightingFunc(X_ij);
          totalWeight += weight;
          totalCount ++;
          totalScore += weight * diff * diff;
          IntStream.range(0, sym_i.dim()).forEach(id -> {
            final double d_i = TRAINING_STEP_COEFF * weight * diff * sym_j.get(id);
            final double d_j = TRAINING_STEP_COEFF * weight * diff * sym_i.get(id);

            sym_i.adjust(id, -d_i / Math.sqrt(softMaxSym_i.get(id)));
            sym_j.adjust(id, -d_j / Math.sqrt(softMaxSym_j.get(id)));

            softMaxSym_i.adjust(id, d_i * d_i);
            softMaxSym_j.adjust(id, d_j * d_j);
          });
          IntStream.range(0, skew_i.dim()).forEach(id -> {
            final double d_i = TRAINING_STEP_COEFF * sign * weight * diff * skew_j.get(id);
            final double d_j = TRAINING_STEP_COEFF * sign * weight * diff * skew_i.get(id);
            final double x_it = skew_i.get(id) - d_i / Math.sqrt(softMaxSkew_i.get(id));
            final double x_jt = skew_j.get(id) - d_j / Math.sqrt(softMaxSkew_j.get(id));

            skew_i.set(id, project(x_it));
            skew_j.set(id, project(x_jt));

            softMaxSkew_i.adjust(id, d_i * d_i);
            softMaxSkew_j.adjust(id, d_j * d_j);
          });
          final double biasStep = TRAINING_STEP_COEFF * weight * diff;

          bias.adjust(i, -biasStep * b_i / Math.sqrt(softMaxBias.get(i)));
          bias.adjust(j, -biasStep * b_j/ Math.sqrt(softMaxBias.get(j)));
          softMaxBias.adjust(i, MathTools.sqr(biasStep * b_i));
          softMaxBias.adjust(j, MathTools.sqr(biasStep * b_j));
        }
        synchronized (counter) {
          counter[0] += totalWeight;
          counter[1] += totalCount;
        }
        return totalScore;
      }).sum();

      Interval.stopAndPrint("Score: " + (score / counter[1]));
    }
    //System.out.println("Likelihood: " + likelihood());
  }

  private double project(double x_t) {
    return x_t > LAMBDA ? x_t - LAMBDA : (x_t < -LAMBDA ? x_t + LAMBDA : 0);
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
