package com.expleague.ml.embedding;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.embedding.exceptions.Word2VecUsageException;
import com.expleague.ml.embedding.model_functions.AbstractModelFunction;
import com.expleague.ml.embedding.text_utils.Vocabulary;

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
    }

    public Vec getVectorByWord(String word) {
        int w = vocabulary.wordToIndex(word);
        if (w == Vocabulary.NO_ENTRY_VALUE)
            throw new Word2VecUsageException("There's no word " + word + " in the vocabulary.");
        return modelVectors.row(w);
    }

    public List<String> getClosestWords(String word, int top) {
        return getClosestWords(getVectorByWord(word), top);
    }

    public List<String> getClosestWords(Vec vector, int top) {
        int[] order = ArrayTools.sequence(0, vocab_size);
        double[] weights = IntStream.of(order).mapToDouble(idx ->
                -VecTools.cosine(modelVectors.row(idx), vector)).toArray();
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

    public int getVocabularySize() {
        return vocab_size;
    }
}
