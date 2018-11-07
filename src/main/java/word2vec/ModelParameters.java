package word2vec;

public class ModelParameters {

    private final String filepath;
    private final int windowSize;
    private final boolean windowSymmetry;
    private final String modelName;

    private ModelParameters(String filepath, String modelName, int windowSize, boolean windowSymmetry) {
        this.filepath = filepath;
        this.windowSize = windowSize;
        this.windowSymmetry = windowSymmetry;
        this.modelName = modelName;
    }

    public String getFilepath() {
        return filepath;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public boolean isWindowSymmetry() {
        return windowSymmetry;
    }

    public String getModelName() {
        return modelName;
    }


    public static class Builder {
        private String filepath = "";
        private int windowSize = 1;
        private boolean windowSymmetry = true;
        private String modelName = "GLOVE";

        public Builder(String filepath) {
            this.filepath = filepath;
        }

        public Builder setModelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder setWindowSize(int window_size) {
            this.windowSize = window_size;
            return this;
        }

        public Builder setWindowSymmetry(boolean window_symmetry) {
            this.windowSymmetry = window_symmetry;
            return this;
        }

        public ModelParameters build() {
            return new ModelParameters(filepath, modelName, windowSize, windowSymmetry);
        }
    }
}
