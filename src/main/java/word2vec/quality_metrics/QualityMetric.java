package word2vec.quality_metrics;

import word2vec.Word2Vec;

public abstract class QualityMetric {

    protected final Word2Vec.Model model;

    public QualityMetric(Word2Vec.Model model) {
        this.model = model;
    }

    public abstract void measure(String input, String output);
}
