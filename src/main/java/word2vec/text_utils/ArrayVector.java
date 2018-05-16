package word2vec.text_utils;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

public class ArrayVector extends ArrayVec {
    public static ArrayVec sumVectors(ArrayVec v1, ArrayVec v2) {
        if (v1.dim() != v2.dim()) {
            throw new RuntimeException("Trying to sum vectors with different dimensions");
        }
        ArrayVec result = new ArrayVec(v1.dim());
        for (int i = 0; i < v1.dim(); i++) {
            result.set(i, v1.get(i) + v2.get(i));
        }
        return result;
    }

    public static ArrayVec vectorsDifference(ArrayVec v1, ArrayVec v2) {
        if (v1.dim() != v2.dim()) {
            throw new RuntimeException("Trying to sum vectors with different dimensions");
        }
        ArrayVec result = new ArrayVec(v1.dim());
        for (int i = 0; i < v1.dim(); i++)
            result.set(i, v1.get(i) - v2.get(i));
        return result;
    }

    public static double countVecNorm(ArrayVec vec) {
        double res = 0d;
        for (int j = 0; j < vec.dim(); j++) {
            final double dv = vec.get(j);
            res += dv * dv;
        }
        return res;
    }

    public static void writeArrayVec(ArrayVec vec, PrintStream fout) throws IOException {
        StringBuilder str = new StringBuilder();
        for (int j = 0; j < vec.dim(); j++) {
            str.append(vec.get(j));
            str.append("\t");
        }
        fout.println(str.toString());
    }

    public static ArrayVec readArrayVec(BufferedReader fin) throws IOException {
        String[] values = fin.readLine().split("\t");
        ArrayVec vec = new ArrayVec(values.length);
        for (int j = 0; j < values.length; j++)
            vec.set(j, Double.parseDouble(values[j]));
        return vec;
    }
}
