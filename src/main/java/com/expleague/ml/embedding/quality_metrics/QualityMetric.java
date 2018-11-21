package com.expleague.ml.embedding.quality_metrics;

import com.expleague.ml.embedding.Model;
import com.expleague.ml.embedding.exceptions.MetricsIOException;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class QualityMetric {

    protected final Model model;
    protected List<List<String>> words = new ArrayList<>();
    protected int words_size = 0;

    public QualityMetric(Model model) {
        this.model = model;
    }

    protected abstract void check(String[] wordsLine, int lineNumber);

    public abstract void measure(String input, String output);

    protected void read(String input) {
        File file = new File(input);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new MetricsIOException("Couldn't find the file to read words from.");
        }
        try {
            words_size = Integer.parseInt(fin.readLine());
            for (int i = 0; i < words_size; i++) {
                String[] line = fin.readLine().split("\t");
                check(line, i);
                words.add(Arrays.asList(line));
            }
            fin.close();
        } catch (IOException e) {
            throw new MetricsIOException("Error occurred during reading from the file.");
        }
    }
}
