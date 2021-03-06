package com.expleague.ml.embedding.model_functions;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.ml.embedding.text_utils.Vocabulary;

import java.io.IOException;

// Model J = sum[ f(Xij) * (viT*uj - logXij)^2]
//TODO stochastic gradient
public abstract class AbstractModelFunction extends FuncC1.Stub {
    final Vocabulary vocab;
    final Mx crcLeft;
    final int vocab_size;

    public AbstractModelFunction(Vocabulary vocab, Mx cooc) {
        this.vocab = vocab;
        this.crcLeft = cooc;
        this.vocab_size = vocab.size();
    }

    public abstract Mx getModelVectors();

    public abstract void trainModel();

    public abstract void saveModel(String filepath) throws IOException;

    public abstract void loadModel(String filepath, int mode) throws IOException;

    public abstract double likelihood();
}
