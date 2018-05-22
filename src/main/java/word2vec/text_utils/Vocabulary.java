package word2vec.text_utils;

import gnu.trove.iterator.TObjectIntIterator;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import word2vec.exceptions.VocabularyBuildingException;

import java.io.*;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

//Immutable vocabulary. Create new vocab if you want to change smth.
//TODO слова с большой буквы то же самое, что с маленькой? А капслок? Просто изменения размера.
public class Vocabulary {

    private List<String> wordsList = new ArrayList<>();
    private TObjectIntMap<String> wordsIndx = new TObjectIntHashMap<>();
    private int size = 0;

    public static int NO_ENTRY_VALUE = -1;
    public static final int MIN_COUNT = 10;

    public Vocabulary(final String filepath) throws VocabularyBuildingException {
        try {
            readWords(filepath);
        } catch (RuntimeException e) {
            //e.printStackTrace();
            final String message = "Constructing vocabulary failed. " + e.getMessage();
            throw new VocabularyBuildingException(message);
        }
    }

    public Vocabulary(List<String> words) {
        size = words.size();
        wordsList.addAll(words);
        for (int i = 0; i < size; i++)
            wordsIndx.put(words.get(i), i);
    }

    public int size() {
        return size;
    }

    public int wordToIndex(String word) {
        if (!wordsIndx.containsKey(word)) {
            return NO_ENTRY_VALUE;
        }
        return wordsIndx.get(word);
    }

    public String indexToWord(int i) {
        if (i < 0 || i >= size) {
            return null;
        }
        return wordsList.get(i);
    }

    public List<String> getEntries() {
        return wordsList;
    }

    private void readWords(final String filepath) {
        File file = new File(filepath);
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Couldn't find the file to construct vocabulary from.");
        }
        String line;
        try {
            TObjectIntMap<String> wordsCount = new TObjectIntHashMap<>();
            while ((line = fin.readLine()) != null) {
                BreakIterator breakIterator = BreakIterator.getWordInstance();
                breakIterator.setText(line);
                int lastIndex = breakIterator.first();
                while (BreakIterator.DONE != lastIndex) {
                    int firstIndex = lastIndex;
                    lastIndex = breakIterator.next();
                    if (lastIndex != BreakIterator.DONE && Character.isLetterOrDigit(line.charAt(firstIndex))) {
                        final String word = line.substring(firstIndex, lastIndex).toLowerCase();
                        wordsCount.adjustOrPutValue(word, 1, 1);
                    }
                }
            }
            for (TObjectIntIterator<String> it = wordsCount.iterator(); it.hasNext();) {
                it.advance();
                if (it.value() >= MIN_COUNT && it.key().length() > 2) {
                    wordsList.add(it.key());
                    wordsIndx.put(it.key(), size);
                    size++;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Couldn't read the vocabulary file.");
        }
    }
}
