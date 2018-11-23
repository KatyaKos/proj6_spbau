package com.expleague.ml.embedding;

import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.embedding.exceptions.Word2VecUsageException;
import com.expleague.ml.embedding.model_functions.AbstractModelFunction;
import com.expleague.ml.embedding.text_utils.Vocabulary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Model {

    private Mx modelVectors;
    private Vocabulary vocabulary;
    private int vocab_size;
    private AbstractModelFunction modelFunction;
    private int vector_size;

    public Model(AbstractModelFunction modelFunction, Vocabulary vocabulary) {
        modelVectors = modelFunction.getModelVectors();
        this.vocabulary = vocabulary;
        vocab_size = vocabulary.size();
        this.modelFunction = modelFunction;
        vector_size = modelVectors.columns();

        normalizeModelVectors();
    }

    private void normalizeModelVectors() {
        final int n = modelVectors.rows();
        IntStream.range(0, n).parallel().forEach(i -> {
            double sq = Math.sqrt(VecTools.sum2(modelVectors.row(i)));
            for (int j = 0; j < modelVectors.columns(); j++) {
                modelVectors.set(i, j, modelVectors.get(i, j) / sq);
            }
        });
    }

    public Vec getVectorByWord(String word) {
        int w = vocabulary.wordToIndex(word);
        if (w == Vocabulary.NO_ENTRY_VALUE)
            throw new Word2VecUsageException("There's no word " + word + " in the vocabulary.");
        return modelVectors.row(w);
    }

    public boolean isWordInVocab(String word) {
        return  vocabulary.containsWord(word);
    }

    public boolean isWordsListInVocab(List<String> words) {
        return vocabulary.containsAll(words);
    }

    public int getIndexByWord(String word) {
        return vocabulary.wordToIndex(word.toLowerCase());
    }

    public List<String> getClosestWords(String word, int top) {
        return getClosestWords(getVectorByWord(word.toLowerCase()), top);
    }

    public List<String> getClosestWords(Vec vector, int top) {
        /*int[] order = ArrayTools.sequence(0, vocab_size);
        double[] weights = IntStream.of(order).mapToDouble(idx ->
                -VecTools.multiply(modelVectors.row(idx), vector)).toArray();
        ArrayTools.parallelSort(weights, order);
        return IntStream.range(0, top).mapToObj(idx ->
                vocabulary.indexToWord(order[idx])).collect(Collectors.toList());*/
        return getClosestWordsExcept(vector, top, new ArrayList<>());
    }

    public List<String> getClosestWordsExcept(Vec vector, int top, List<String> exceptWords) {
        int[] order = ArrayTools.sequence(0, vocab_size);
        List<Integer> exceptIds = new ArrayList<>();
        exceptWords.forEach(word -> exceptIds.add(vocabulary.wordToIndex(word)));
        double[] weights = IntStream.of(order).mapToDouble(idx -> {
            if (exceptIds.contains(idx))
                return Double.MAX_VALUE;
            return -VecTools.multiply(modelVectors.row(idx), vector);
        }).toArray();
        ArrayTools.parallelSort(weights, order);
        return IntStream.range(0, top).mapToObj(idx ->
                vocabulary.indexToWord(order[idx])).collect(Collectors.toList());
    }

    public double countLikelihood() {
        return modelFunction.likelihood();
    }

    public int getVectorSize() {
        return vector_size;
    }

    public List<String> getVocabulary() {
        return vocabulary.getEntries();
    }
}
