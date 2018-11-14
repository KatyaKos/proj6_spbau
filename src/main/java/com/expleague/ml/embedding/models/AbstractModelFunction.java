package com.expleague.ml.embedding.models;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.embedding.text_utils.Vocabulary;

import java.io.IOException;
import java.util.List;

// Model J = sum[ f(Xij) * (viT*uj - logXij)^2]
//TODO stochastic gradient
public abstract class AbstractModelFunction extends FuncC1.Stub {
    final Vocabulary vocab;
    final Mx crcLeft;
    final Mx crcRight;
    final int vocab_size;

    public AbstractModelFunction(Vocabulary vocab, Mx cooc) {
        this.vocab = vocab;
        this.crcLeft = cooc;
        this.crcRight = MxTools.transpose(crcLeft);
        this.vocab_size = vocab.size();
    }

    public abstract void prepareReadyModel();

    public abstract void trainModel();

    public abstract void saveModel(String filepath) throws IOException;

    public abstract void loadModel(String filepath) throws IOException;

    public abstract Vec getVectorByWord(String word);

    public abstract List<String> getWordByVector(Vec vector);

    public abstract double likelihood();

    public abstract double getDistance(String from, String to);

    public abstract double getSkewVector(String word);
}
