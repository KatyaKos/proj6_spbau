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

public class GloveModelFunction extends AbstractModelFunction {
  final private static int TRAINING_ITERS = 100;
  final private static double TRAINING_STEP_COEFF = 0.1;

  private final static double WEIGHTING_X_MAX = 100;
  private final static double WEIGHTING_ALPHA = 0.75;
  private final static int VECTOR_SIZE = 50;

  private Mx leftVectors;
  private Mx rightVectors;


  public GloveModelFunction(Vocabulary voc, Mx coocc) {
      super(voc, coocc);
  }

  private void initialize() {
    leftVectors = new VecBasedMx(vocab_size, VECTOR_SIZE);
    rightVectors = new VecBasedMx(vocab_size, VECTOR_SIZE);
    for (int i = 0; i < vocab_size; i++) {
      for (int j = 0; j < VECTOR_SIZE; j++) {
        leftVectors.set(i, j, Math.random());
        rightVectors.set(i, j, Math.random());
      }
    }
  }

  @Override
  public Mx getModelVectors() {
    return leftVectors;
  }

  @Override
  public double likelihood() {
    long[] totalComponents = new long[]{0};
    double total = IntStream.range(0, crcLeft.rows()).parallel().mapToDouble(i -> {
      final VecIterator nz = crcLeft.row(i).nonZeroes();
      double res = 0;
      int counter = 0;
      while (nz.advance()) {
        counter++;
        final int j = nz.index();
        final double X_ij = nz.value();
        res += weightingFunc(X_ij) * MathTools.sqr(VecTools.multiply(leftVectors.row(i), rightVectors.row(j)) - Math.log(1d + X_ij));
      }
      synchronized (totalComponents) {
        totalComponents[0] += counter;
      }
      return res;
    }).sum();
    return total / totalComponents[0];
  }

  @Override
  public void saveModel(String filepath) throws IOException {
    File file = new File(filepath + "/vectors.txt");
    File file_sum = new File(filepath + "/left+right.txt");
    PrintStream fout = new PrintStream(file);
    PrintStream fout_sum = new PrintStream(file_sum);
    fout.println("GLOVE");
    for (int i = 0; i < vocab_size; i++) {
      VecIO.writeVec(fout, leftVectors.row(i));
      VecIO.writeVec(fout_sum, VecTools.sum(leftVectors.row(i), rightVectors.row(i)));
    }
    for (int i = 0; i < vocab_size; i++) {
      VecIO.writeVec(fout, rightVectors.row(i));
    }
    fout.close();
    fout_sum.close();
  }

  @Override
  public void loadModel(String filepath, int mode) throws IOException {
    if (mode == 0) filepath += "/vectors.txt";
    else if (mode == 1) filepath += "/left+right.txt";
    try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath)))) {
      if (mode == 0) {
        fin.readLine();
        leftVectors = VecIO.readMx(fin, vocab_size);
        rightVectors = VecIO.readMx(fin, vocab_size);
      } else if (mode == 1){
        leftVectors = VecIO.readMx(fin, vocab_size);
      }
    }
    catch (FileNotFoundException e) {
      throw new LoadingModelException("Couldn't find vectors file to load the model from.");
    }
  }

  private double weightingFunc(double x) {
    return x < WEIGHTING_X_MAX ? Math.pow((x / WEIGHTING_X_MAX), WEIGHTING_ALPHA) : 1;
  }

  @Override
  public void trainModel() {
    if (leftVectors == null) {
        initialize();
    }
    final Vec bias = new ArrayVec(rightVectors.rows());

    final Mx softMaxLeft = new VecBasedMx(leftVectors.rows(), leftVectors.columns());
    final Mx softMaxRight = new VecBasedMx(rightVectors.rows(), rightVectors.columns());
    final Vec softMaxBias = new ArrayVec(rightVectors.rows());

    VecTools.fill(softMaxLeft, 1);
    VecTools.fill(softMaxRight, 1);
    VecTools.fill(softMaxBias, 1);
    for (int iter = 0; iter < TRAINING_ITERS; iter++) {
      Interval.start();
      double[] counter = new double[]{0, 0};
      double score = IntStream.range(0, crcLeft.rows()).parallel().mapToDouble(i -> { // left part derivative
        final VecIterator nz = crcLeft.row(i).nonZeroes();
        final Vec left = leftVectors.row(i);
        final Vec softMax = softMaxLeft.row(i);
        double totalScore = 0;
        double totalCount = 0;
        double totalWeight = 0;
        while (nz.advance()) {
          final int j = nz.index();
          final Vec right = rightVectors.row(j);
          final double X_ij = nz.value();
          final double asum = VecTools.multiply(left, right);
          final double diff = bias.get(i) + bias.get(j) + asum - Math.log(1d + X_ij);
          final double weight = weightingFunc(X_ij);

          totalWeight += weight;
          totalScore += weight * MathTools.sqr(diff);
          IntStream.range(0, right.dim()).forEach(id -> {
            final double d = TRAINING_STEP_COEFF * weight * diff * right.get(id);
            left.adjust(id, -d / Math.sqrt(softMax.get(id)));
            softMax.adjust(id, d * d);
          });
          final double biasStep = TRAINING_STEP_COEFF * weight * diff;

          bias.adjust(i, -biasStep / Math.sqrt(softMaxBias.get(i)));
          softMaxBias.adjust(i, MathTools.sqr(biasStep));
          totalCount++;
        }
        synchronized (counter) {
          counter[0] += totalWeight;
          counter[1] += totalCount;
        }
//        VecTools.assign(leftVectors.row(i), left);
        return totalScore;
      }).sum();
      IntStream.range(0, crcRight.rows()).parallel().forEach(j -> { // right part derivative
        final VecIterator nz = crcRight.row(j).nonZeroes();
        final Vec right = rightVectors.row(j);
        final Vec softMax = softMaxRight.row(j);
        while (nz.advance()) {
          int i = nz.index();
          if (i == j)
            continue;
          final Vec left = leftVectors.row(i);
          final double X_ij = nz.value();
          final double asum = VecTools.multiply(left, rightVectors.row(j));
          final double diff = bias.get(i) + bias.get(j) + asum - Math.log(1d + X_ij);
          final double weight = weightingFunc(X_ij);
          IntStream.range(0, left.dim()).forEach(id -> {
            final double d = TRAINING_STEP_COEFF * weight * diff * left.get(id);
            right.adjust(id, -d / Math.sqrt(softMax.get(id)));
            softMax.adjust(id, d * d);
          });

          final double biasStep = TRAINING_STEP_COEFF * weight * diff;
          bias.adjust(j, -biasStep / Math.sqrt(softMaxBias.get(j)));
          softMaxBias.adjust(j, MathTools.sqr(biasStep));
        }
//        VecTools.assign(rightVectors.row(j), right);
      });

      Interval.stopAndPrint("Iteration " + iter + ", Score " + score / counter[1]);
    }
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
