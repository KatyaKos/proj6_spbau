package com.expleague.ml.embedding.quality_metrics.impl;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.ml.embedding.Model;
import com.expleague.ml.embedding.exceptions.MetricsIOException;
import com.expleague.ml.embedding.quality_metrics.QualityMetric;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class ArithmeticMetric extends QualityMetric {

    private int vector_size;

    public ArithmeticMetric(Model model) {
        super(model);
        vector_size = model.getVectorSize();
    }

    @Override
    protected void check(String[] wordsLine, int lineNumber) {
        if (wordsLine.length != 4) throw new MetricsIOException("There should be four words in each line." +
                String.format(" Error occurred in line number %d.", lineNumber + 1));
    }

    @Override
    public void measure(String input, String output) {
        read(input);

        File file = new File(output);
        PrintStream fout;
        try {
            fout = new PrintStream(file);
        } catch (FileNotFoundException e) {
            throw new MetricsIOException("Couldn't find the file to write the closer-further metrics results to");
        }

        Mx predictedVectors = new VecBasedMx(words_size, vector_size);
        IntStream.range(0, words_size).forEach(i -> {
            Vec v1 = model.getVectorByWord(words.get(i).get(0));
            Vec v2 = model.getVectorByWord(words.get(i).get(1));
            Vec v3 = model.getVectorByWord(words.get(i).get(1));
            IntStream.range(0, vector_size).forEach(j -> {
                predictedVectors.set(i, j, v1.get(j));
                predictedVectors.adjust(i, j, -v2.get(j));
                predictedVectors.adjust(i, j, v3.get(j));
            });
        });

        List<List<String>> results = new ArrayList<>();
        final int[] successes = {0};
        IntStream.range(0, words_size).forEach(i -> {
            List<String> result = model.getClosestWords(predictedVectors.row(i), 5);
            if (result.contains(words.get(i).get(3))) successes[0]++;
        });

        fout.println(String.format("Number of successes is %d out of %d", successes[0], words_size));
        for (int i = 0; i < words_size; i++)
            fout.println(String.format("%s\t->\t%s", words.get(i).get(3),
                    String.join(", ", results.get(i))));
        fout.close();
    }

}
