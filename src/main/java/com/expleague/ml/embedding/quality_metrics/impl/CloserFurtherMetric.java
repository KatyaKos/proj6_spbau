package com.expleague.ml.embedding.quality_metrics.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.embedding.Model;
import com.expleague.ml.embedding.exceptions.MetricsIOException;
import com.expleague.ml.embedding.quality_metrics.QualityMetric;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class CloserFurtherMetric extends QualityMetric {

    public CloserFurtherMetric(Model model) {
        super(model);
    }

    @Override
    protected void check(List<String> wordsLine, int lineNumber) {
        if (wordsLine.size() != 3) throw new MetricsIOException("There should be three words in each line." +
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

            List<String> result = new ArrayList<>(words_size);
            int success = countMetric(result);
            fout.println(String.format("%d successes out of %d", success, words_size));
            fout.println();
            for (int i = 0; i < words_size; i++) {
                fout.println(result.get(i));
            }
            fout.close();
        }
    }

    private int countMetric(List<String> result) {
        int success = 0;
        for (int i = 0; i < words_size; i++) {
            if (model.isWordsListInVocab(words.get(i))) {
                final String w1 = words.get(i).get(0);
                final String w2 = words.get(i).get(1);
                final String w3 = words.get(i).get(2);
                final Vec v1 = model.getVectorByWord(w1);
                final Vec v2 = model.getVectorByWord(w2);
                final Vec v3 = model.getVectorByWord(w3);
                final boolean suc = model.getDistance(v1, v2) > model.getDistance(v1, v3);
                result.add(resultToString(suc, w1, w2, w3));
                if (suc) success++;
            } else {
                List<String> excludes = new ArrayList<>();
                for (String word : words.get(i)) {
                    if (!model.isWordInVocab(word))
                        excludes.add(word);
                }
                result.add("WORDS " + String.join(", ", excludes) + " ARE NOT IN VOCABULARY!");
            }
        }
        return success;
    }

    private String resultToString(boolean res, String w1, String w2, String w3) {
        if (res) {
            return String.format("TRUE: \tWord \'%s\' is closer to \'%s\' than to word \'%s\'.", w1, w2, w3);
        } else {
            return String.format("FALSE:\tWord \'%s\' is closer to \'%s\' than to word \'%s\'.", w1, w3, w2);
        }
    }
}
