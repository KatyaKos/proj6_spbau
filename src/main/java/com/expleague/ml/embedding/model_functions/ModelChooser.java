package com.expleague.ml.embedding.model_functions;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.ml.embedding.ModelParameters;
import com.expleague.ml.embedding.text_utils.Vocabulary;

public class ModelChooser {

    // GLOVE -> GloveModelFunction
    // DECOMP -> DecomposingGloveModel

    public static AbstractModelFunction model(ModelParameters modelParameters, Vocabulary vocab, Mx crcs) {
        switch (modelParameters.getModelName()) {
            case "GLOVE": return new GloveModelFunction(vocab, crcs,
                    modelParameters.getGloveVecSize(), modelParameters.getTrainingIters());
            case "DECOMP": return new DecomposingGloveModelFunction(vocab, crcs,
                    modelParameters.getSymSize(), modelParameters.getSkewSize(), modelParameters.getTrainingIters());
            default: return new GloveModelFunction(vocab, crcs,
                    modelParameters.getGloveVecSize(), modelParameters.getTrainingIters());
        }
    }
}
