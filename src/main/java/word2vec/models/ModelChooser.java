package word2vec.models;

import com.expleague.commons.math.vectors.Mx;
import word2vec.text_utils.CooccurencesBuilder;
import word2vec.text_utils.Vocabulary;

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
