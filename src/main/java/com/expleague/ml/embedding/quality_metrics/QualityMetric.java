package word2vec.quality_metrics;

import com.expleague.ml.embedding.Word2Vec;

public abstract class QualityMetric {

    protected final Word2Vec.Model model;

    public QualityMetric(Word2Vec.Model model) {
        this.model = model;
    }

    public abstract void measure(String input, String output);
}
