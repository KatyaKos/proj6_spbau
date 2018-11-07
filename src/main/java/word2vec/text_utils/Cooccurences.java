package word2vec.text_utils;


import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.linked.TIntLinkedList;
import word2vec.ModelParameters;
import word2vec.exceptions.CooccurencesBuildingException;

import java.io.*;
import java.text.BreakIterator;

// Table X = {Xij}, Xij = number of times word j occures in the context of the word i.
// If not symmetric - only looks at word's left neighbours.
public class Cooccurences {

    private SparseVec[] crcs;
    private int vocab_size = 0;

    private int WINDOW_SIZE;
    private boolean SYMMETRIC;

    public Cooccurences(int vocab_size, int window, boolean symmetry, SparseVec[] coocures) {
        this.WINDOW_SIZE = window;
        this.SYMMETRIC = symmetry;
        this.vocab_size = vocab_size;
        this.crcs = coocures;
    }

    public Cooccurences(Vocabulary vocab, ModelParameters modelParameters) throws CooccurencesBuildingException {
        this.WINDOW_SIZE = modelParameters.getWindowSize();
        this.SYMMETRIC = modelParameters.isWindowSymmetry();
        this.vocab_size = vocab.size();

        crcs = new SparseVec[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            crcs[i] = new SparseVec(vocab_size);
            for (int j = 0; j < vocab_size; j++) {
                crcs[i].set(j, 0d);
            }
        }

        try {
            count_cooccur(modelParameters.getFilepath(), vocab);
        } catch (RuntimeException e) {
            //e.printStackTrace();
            final String message = "Constructing coocurences table failed." + e.getMessage();
            throw new CooccurencesBuildingException(message);
        }
    }

    public int getWindowSize() {
        return WINDOW_SIZE;
    }

    public boolean getSymmetric() {
        return SYMMETRIC;
    }

    public double getValue(int i, int j) {
        if (i < 0 || i > vocab_size || j < 0 || j > vocab_size) {
            throw new RuntimeException("Trying to acces word [" + String.valueOf(i) + "," +
                    String.valueOf(j) + "] in the cooccurences table of size " + String.valueOf(vocab_size));
        }
        return crcs[i].get(j);
    }

    private void count_cooccur(String filepath, Vocabulary vocab) throws RuntimeException {
        TIntLinkedList queue = new TIntLinkedList() {
            @Override
            public boolean add(int val) {
                if(this.size() < WINDOW_SIZE)
                    super.add(val);
                else
                {
                    super.removeAt(0);
                    super.add(val);
                }
                return true;
            }
        };

        File file = new File(filepath);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Couldn't find the file to construct cooccurences table from.");
        }
        String line;
        try {
            while ((line = fin.readLine()) != null) {
                BreakIterator breakIterator = BreakIterator.getWordInstance();
                breakIterator.setText(line);
                int lastIndex = breakIterator.first();
                while (BreakIterator.DONE != lastIndex) {
                    int firstIndex = lastIndex;
                    lastIndex = breakIterator.next();
                    if (lastIndex != BreakIterator.DONE && Character.isLetterOrDigit(line.charAt(firstIndex))) {
                        final String word = line.substring(firstIndex, lastIndex);
                        int w1 = vocab.wordToIndex(word);
                        if (w1 == Vocabulary.NO_ENTRY_VALUE) {
                            continue;
                        }
                        TIntIterator iter = queue.iterator();
                        while (iter.hasNext()) {
                            int w2 = iter.next();
                            crcs[w2].adjust(w1, 1d);
                            if (SYMMETRIC) {
                                crcs[w1].adjust(w2, 1d);
                            }
                        }
                        queue.add(w1);
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Couldn't read the file.");
        }
    }
}