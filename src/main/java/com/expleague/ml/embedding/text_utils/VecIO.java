package com.expleague.ml.embedding.text_utils;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;

public class VecIO {
    public static void writeVec(Writer fout, Vec vec) throws IOException {
        fout.append(String.valueOf(vec.get(0)));
        for (int i = 1; i < vec.dim(); i++){
            fout.append(" ");
            fout.append(Double.toString(vec.get(i)));
        }
    }

    public static void readVecTo(BufferedReader fin, Vec to) throws IOException {
        String[] values = fin.readLine().split(" ");
        for (int j = 0; j < values.length; j++)
            to.set(j, Double.parseDouble(values[j]));
    }

    public static Vec readVec(BufferedReader fin) throws IOException {
        String[] values = fin.readLine().split(" ");
        Vec to = new ArrayVec(values.length);
        for (int j = 0; j < values.length; j++)
            to.set(j, Double.parseDouble(values[j]));
        return to;
    }

    public static Mx readMx(BufferedReader fin, int rows) throws IOException {
        final Vec vec0 = readVec(fin);
        int columns = vec0.dim();
        Mx to = new VecBasedMx(rows, columns);
        VecTools.assign(to.row(0), vec0);
        for (int i = 1; i < rows; i++) {
            final Vec vec = readVec(fin);
            VecTools.assign(to.row(i), vec);
        }
        return to;
    }
}
