package com.expleague.ml.embedding.quality_metrics.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.embedding.Model;
import com.expleague.ml.embedding.exceptions.MetricsIOException;
import com.expleague.ml.embedding.exceptions.Word2VecUsageException;
import com.expleague.ml.embedding.quality_metrics.QualityMetric;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class ArithmeticMetric extends QualityMetric {

    private int vector_size;

    public ArithmeticMetric(Model model) {
        super(model);
        vector_size = model.getVectorSize();
    }

    @Override
    protected void check(List<String> wordsLine, int lineNumber) {
        if (wordsLine.size() != 4) throw new MetricsIOException("There should be four words in each line." +
                String.format(" Error occurred in line number %d.", lineNumber + 1));
    }

    @Override
    public void measure(String input, String output) {
        readMetricsNames(input);

        for (String fileName : files) {
            String short_name = readMetricsFile(fileName);
            System.out.println("Started working with " + short_name);
            File file = new File(output + "/" + short_name);
            PrintStream fout;
            try {
                fout = new PrintStream(file);
            } catch (FileNotFoundException e) {
                throw new MetricsIOException("Couldn't find the file to write the closer-further metrics results to");
            }

            List<String> results = new ArrayList<>();
            final int[] counts = {0, 0, 0}; // 0 for total, 1 for top1, 2 for top5
            countMetric(results, counts);

            fout.println(String.format("Number of top1 is %d out of %d (%d%%)",
                    counts[1], counts[0], 100 * counts[1] / counts[0]));
            fout.println(String.format("Number of top5 is %d out of %d (%d%%)",
                    counts[2], counts[0], 100 * counts[2] / counts[0]));
            for (int i = 0; i < words_size; i++) {
                fout.println(String.format("%s\t->\t%s", words.get(i).get(3), results.get(i)));
            }
            fout.close();
        }
    }

    private void countMetric(List<String> results, final int[] counts) {
        IntStream.range(0, words_size).forEach(i -> {
            Vec predicted = new ArrayVec(vector_size);
            if (model.isWordsListInVocab(words.get(i))) {
                Vec v1 = model.getVectorByWord(words.get(i).get(0));
                Vec v2 = model.getVectorByWord(words.get(i).get(1));
                Vec v3 = model.getVectorByWord(words.get(i).get(2));
                IntStream.range(0, vector_size).forEach(j -> {
                    predicted.set(j, v2.get(j));
                    predicted.adjust(j, -v1.get(j));
                    predicted.adjust(j, v3.get(j));
                });

                List<String> result = model.getClosestWordsExcept(predicted, 5, words.get(i).subList(0, 3));
                results.add(String.join(", ", result));
                if (result.contains(words.get(i).get(3))) counts[2]++;
                if (words.get(i).get(3).equals(result.get(0))) counts[1]++;
                counts[0]++;
            } else {
                List<String> excludes = new ArrayList<>();
                for (String word : words.get(i)) {
                    if (!model.isWordInVocab(word))
                        excludes.add(word);
                }
                results.add("WORDS " + String.join(", ", excludes) + " ARE NOT IN VOCABULARY!");
            }
        });
    }

}
