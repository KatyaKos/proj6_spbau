package word2vec.quality_metrics.impl;

import word2vec.Word2Vec;
import word2vec.exceptions.MetricsIOException;
import word2vec.quality_metrics.QualityMetric;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CloserFurtherMetric extends QualityMetric {

    private int size = 0;
    private List<String> leadWords = new ArrayList<>();
    private List<String> closerWords = new ArrayList<>();
    private List<String> furtherWords = new ArrayList<>();


    public CloserFurtherMetric(Word2Vec.Model model) {
        super(model);
    }

    @Override
    public void measure(String input, String output) {
        readTriples(input);
        boolean[] result = new boolean[size];
        int success = 0;
        for (int i = 0; i < size; i++) {
            result[i] = model.isCloser(leadWords.get(i), closerWords.get(i), furtherWords.get(i));
            if (result[i]) success++;
        }
        writeTriplesResult(output, result, success);
    }

    private void readTriples(String input) {
        File file = new File(input);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new MetricsIOException("Couldn't find the file to read closer-further metrics words from.");
        }
        try {
            size = Integer.parseInt(fin.readLine());
            for (int i = 0; i < size; i++) {
                String[] words = fin.readLine().split("\t");
                if (words.length != 3) throw new MetricsIOException("There should be three words in each line." +
                        String.format(" Error occurred in line number %d.", i + 1));
                leadWords.add(words[0]);
                closerWords.add(words[1]);
                furtherWords.add(words[2]);
            }
            fin.close();
        } catch (IOException e) {
            throw new MetricsIOException("Error occurred during reading from the file with closer-further metrics words.");
        }
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
