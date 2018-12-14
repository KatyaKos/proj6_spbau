package com.expleague.ml.embedding;

public class ModelParameters {
    private final String filepath;
    private final int leftWindow;
    private final int rightWindow;
    private final String modelName;

    private ModelParameters(String filepath, String modelName, int leftWindow, int rightWindow) {
        this.filepath = filepath;
        this.leftWindow = leftWindow;
        this.rightWindow = rightWindow;
        this.modelName = modelName;
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


    public static class Builder {
        private String filepath = "";
        private int leftWindow = 15;
        private int rightWindow = 15;
        private String modelName = "GLOVE";

        public Builder(String filepath) {
            this.filepath = filepath;
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

        public ModelParameters build() {
            return new ModelParameters(filepath, modelName, leftWindow, rightWindow);
        }
    }
}
