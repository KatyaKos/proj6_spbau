package com.expleague.ml.embedding;

public class ModelParameters {
    private final String filepath;
    private final int leftWindow;
    private final int rightWindow;
    private final String modelName;
    private final int gloveVecSize;
    private final int symSize;
    private final int skewSize;
    private final int trainingIters;

    private ModelParameters(String filepath, String modelName, int leftWindow, int rightWindow,
                            int gloveVecSize, int symSize, int skewSize, int trainingIters) {
        this.filepath = filepath;
        this.leftWindow = leftWindow;
        this.rightWindow = rightWindow;
        this.modelName = modelName;
        this.gloveVecSize = gloveVecSize;
        this.symSize = symSize;
        this. skewSize = skewSize;
        this.trainingIters = trainingIters;
    }

    public String getFilepath() {
        return filepath;
    }

    public int getLeftWindow() {
        return leftWindow;
    }

    public int getRightWindow() {
        return rightWindow;
    }

    public String getModelName() {
        return modelName;
    }

    public int getGloveVecSize() {
        return gloveVecSize;
    }

    public int getSymSize() {
        return symSize;
    }

    public int getSkewSize() {
        return skewSize;
    }

    public int getTrainingIters() {
        return trainingIters;
    }


    public static class Builder {
        private String filepath = "";
        private int leftWindow = 15;
        private int rightWindow = 15;
        private String modelName = "GLOVE";
        private int gloveVecSize = 50;
        private int symSize = 50;
        private int skewSize = 10;
        private int trainingIters = 25;

        public Builder(String filepath) {
            this.filepath = filepath;
        }


        public ModelParameters build() {
            return new ModelParameters(filepath, modelName, leftWindow, rightWindow,
                    gloveVecSize, symSize, skewSize, trainingIters);
        }

        public Builder setModelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder setLeftWindow(int leftWindow) {
            this.leftWindow = leftWindow;
            return this;
        }

        public Builder setRightWindow(int rightWindow) {
            this.rightWindow = rightWindow;
            return this;
        }

        public Builder setGloveVecSize(int gloveVecSize) {
            this.gloveVecSize = gloveVecSize;
            return this;
        }

        public Builder setSymSize(int symSize) {
            this.symSize = symSize;
            return this;
        }

        public Builder setSkewSize(int skewSize) {
            this.skewSize = skewSize;
            return this;
        }

        public Builder setTrainingIters(int trainingIters) {
            this.trainingIters = trainingIters;
            return this;
        }
    }
}
