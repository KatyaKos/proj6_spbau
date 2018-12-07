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
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.IntStream;

public class GloveModelFunction extends AbstractModelFunction {
  final private static int TRAINING_ITERS = 25;
  final private static double TRAINING_STEP_COEFF = 0.05;

  private final static double WEIGHTING_X_MAX = 10;
  private final static double WEIGHTING_ALPHA = 0.75;
  private final static int VECTOR_SIZE = 50;

  private Mx leftVectors;
  private Mx rightVectors;


  public GloveModelFunction(Vocabulary voc, Mx coocc) {
      super(voc, coocc);
  }

  private void initialize() {
    leftVectors = new VecBasedMx(vocab_size, VECTOR_SIZE + 1);
    rightVectors = new VecBasedMx(vocab_size, VECTOR_SIZE + 1);
    for (int i = 0; i < vocab_size; i++) {
      for (int j = 0; j <= VECTOR_SIZE; j++) {
        leftVectors.set(i, j, initializeValue());
        rightVectors.set(i, j, initializeValue());
      }
    }
  }

  private double initializeValue() {
    return (Math.random() - 0.5) / VECTOR_SIZE;
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
    try (Writer fout = Files.newBufferedWriter(Paths.get(filepath + "/vectors.txt"))) {
      fout.append("GLOVE\n");
      for (int i = 0; i < vocab_size; i++) {
        VecIO.writeVec(fout, leftVectors.row(i));
        fout.append('\n');
      }
      for (int i = 0; i < vocab_size; i++) {
        VecIO.writeVec(fout, rightVectors.row(i));
        fout.append('\n');
      }
    }

    try (Writer fout = Files.newBufferedWriter(Paths.get(filepath + "/left+right.txt"))) {
      for (int i = 0; i < vocab_size; i++) {
        VecIO.writeVec(fout, VecTools.sum(leftVectors.row(i), rightVectors.row(i)));
        fout.append('\n');
      }
    }
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

    final Mx softMaxLeft = new VecBasedMx(leftVectors.rows(), leftVectors.columns());
    final Mx softMaxRight = new VecBasedMx(rightVectors.rows(), rightVectors.columns());
    VecTools.fill(softMaxLeft, 1.);
    VecTools.fill(softMaxRight, 1.);

    for (int iter = 0; iter < TRAINING_ITERS; iter++) {
      Interval.start();
      double[] counter = new double[]{0, 0};
      double score = IntStream.range(0, vocab_size).parallel().mapToDouble(i -> {
        final VecIterator nz = crcLeft.row(i).nonZeroes();
        final Vec left = leftVectors.row(i);
        final Vec softMaxL = softMaxLeft.row(i);
        double totalScore = 0;
        double totalCount = 0;
        double totalWeight = 0;
        while (nz.advance()) {
          final int j = nz.index();
          final Vec right = rightVectors.row(j);
          final Vec softMaxR = softMaxRight.row(j);
          final double X_ij = nz.value();
          final double asum = VecTools.multiply(left, right) - left.get(VECTOR_SIZE) * right.get(VECTOR_SIZE);
          final double diff = left.get(VECTOR_SIZE) + right.get(VECTOR_SIZE) + asum - Math.log(1d + X_ij);
          final double weight = weightingFunc(X_ij);
          final double fdiff = TRAINING_STEP_COEFF * diff * weight;

          totalWeight += weight;
          totalScore += 0.5 * weight * MathTools.sqr(diff);

          IntStream.range(0, VECTOR_SIZE).forEach(id -> {
            final double dL = fdiff * right.get(id);
            final double dR = fdiff * left.get(id);
            left.adjust(id, -dL / Math.sqrt(softMaxL.get(id)));
            right.adjust(id, -dR / Math.sqrt(softMaxR.get(id)));
            softMaxL.adjust(id, dL * dL);
            softMaxR.adjust(id, dR * dR);
          });

          left.adjust(VECTOR_SIZE, -fdiff / Math.sqrt(softMaxL.get(VECTOR_SIZE)));
          right.adjust(VECTOR_SIZE, -fdiff / Math.sqrt(softMaxR.get(VECTOR_SIZE)));
          softMaxL.adjust(VECTOR_SIZE, MathTools.sqr(fdiff));
          softMaxR.adjust(VECTOR_SIZE, MathTools.sqr(fdiff));
          totalCount++;
        }
        synchronized (counter) {
          counter[0] += totalWeight;
          counter[1] += totalCount;
        }
        return totalScore;
      }).sum();

      Interval.stopAndPrint("Iteration " + iter + ", Score " + score / counter[1] + ", Total Score " + score + ", Count " + counter[1]);
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
