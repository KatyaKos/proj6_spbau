package com.expleague.ml.embedding.quality_metrics.impl;

import com.expleague.ml.embedding.Model;
import com.expleague.ml.embedding.exceptions.MetricsIOException;
import com.expleague.ml.embedding.quality_metrics.QualityMetric;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CloserFurtherMetric extends QualityMetric {

    private int size = 0;
    private List<String> leadWords = new ArrayList<>();
    private List<String> closerWords = new ArrayList<>();
    private List<String> furtherWords = new ArrayList<>();


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
        readMetricsFile(input);
        boolean[] result = new boolean[size];
        int success = 0;
        for (int i = 0; i < size; i++) {
            //result[i] = model.isCloser(leadWords.get(i), closerWords.get(i), furtherWords.get(i));
            if (result[i]) success++;
        }
        writeTriplesResult(output, result, success);
    }


    private void writeTriplesResult(String output, boolean[] result, int success) {
        File file = new File(output);
        PrintStream fout;
        try {
            fout = new PrintStream(file);
        } catch (FileNotFoundException e) {
            throw new MetricsIOException("Couldn't find the file to write the closer-further metrics results to");
        }
        fout.println(String.format("%d successes out of %d", success, size));
        fout.println();
        for (int i = 0; i < size; i++) {
            fout.println(String.format("%s\tWord \'%s\' is closer to \'%s\' than word \'%s\'.",
                    resultToString(result[i]), closerWords.get(i), leadWords.get(i), furtherWords.get(i)));
        }
        fout.close();
    }

    private String resultToString(boolean res) {
        if (res) return "SUCCESS";
        else return "FAIL";
    }
}
