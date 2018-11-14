package com.expleague.ml.embedding.models;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.ml.embedding.text_utils.Vocabulary;

public class ModelChooser {

    // GLOVE -> GloveModelFunction
    // DECOMP -> DecomposingGloveModel

    public static AbstractModelFunction model(String name, Vocabulary vocab, Mx crcs) {
        switch (name) {
            case "GLOVE": return new GloveModelFunction(vocab, crcs);
            case "DECOMP": return new DecomposingGloveModelFunction(vocab, crcs);
            default: return new GloveModelFunction(vocab, crcs);
        }
    }
}
