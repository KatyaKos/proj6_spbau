package com.expleague.ml.embedding.quality_metrics;

import com.expleague.ml.embedding.Model;
import com.expleague.ml.embedding.exceptions.MetricsIOException;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public abstract class QualityMetric {

    protected final Model model;
    protected List<List<String>> words = new ArrayList<>();
    protected List<String> files = new ArrayList<>();
    protected int words_size = 0;

    public QualityMetric(Model model) {
        this.model = model;
    }

    protected abstract void check(List<String> wordsLine, int lineNumber);

    public abstract void measure(String input, String output);

    private String normalizeWord(String input) {
        return input.toLowerCase();
    }

    protected void readMetricsNames(String fileName) {
        File file = new File(fileName);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new MetricsIOException("Couldn't find the file with names of metrics files.");
        }
        try {
            int num = Integer.parseInt(fin.readLine());
            for (int i = 0; i < num; i++)
                files.add(fin.readLine());
            fin.close();
        } catch (IOException e) {
            throw new MetricsIOException("Error occurred during reading from the file.");
        }
    }

    protected String readMetricsFile(String input) {
        words = new ArrayList<>();
        File file = new File(input);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new MetricsIOException("Couldn't find the file to readMetricsFile words from.");
        }
        try {
            words_size = Integer.parseInt(fin.readLine());
            for (int i = 0; i < words_size; i++) {
                List<String> line = Arrays.asList(fin.readLine().split(" "));
                check(line, i);
                IntStream.range(0, line.size()).forEach(id -> line.set(id, normalizeWord(line.get(id))));
                words.add(line);
            }
            fin.close();
        } catch (IOException e) {
            throw new MetricsIOException("Error occurred during reading from the file.");
        }
        return file.getName();
    }
}
