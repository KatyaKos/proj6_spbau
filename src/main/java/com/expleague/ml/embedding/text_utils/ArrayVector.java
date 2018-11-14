package com.expleague.ml.embedding.text_utils;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

public class ArrayVector extends ArrayVec {

    public static void writeArrayVec(Vec vec, PrintStream fout) throws IOException {
        StringBuilder str = new StringBuilder();
        for (int j = 0; j < vec.dim(); j++) {
            str.append(vec.get(j));
            str.append("\t");
        }
        fout.println(str.toString());
    }

    public static Vec readArrayVec(BufferedReader fin) throws IOException {
        String[] values = fin.readLine().split("\t");
        Vec vec = new ArrayVec(values.length);
        for (int j = 0; j < values.length; j++)
            vec.set(j, Double.parseDouble(values[j]));
        return vec;
    }
}
