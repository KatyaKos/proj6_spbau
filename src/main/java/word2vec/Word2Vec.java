package word2vec;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import word2vec.exceptions.*;
import word2vec.models.AbstractModelFunction;
import word2vec.models.ModelChooser;
import word2vec.text_utils.Cooccurences;
import word2vec.text_utils.Vocabulary;
import word2vec.text_utils.ArrayVector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Word2Vec {

    private final static String DEFAULT_MODEL = "GLOVE";

    private Vocabulary vocabulary = null;
    private int vocab_size = 0;
    private Cooccurences cooccurences = null;
    private AbstractModelFunction model = null;

    public List<String> vocab() throws EmptyVocabularyException {
        if (vocabulary == null) {
            throw new EmptyVocabularyException("Build vocabulary first.");
        }
        return vocabulary.getEntries();
    }

    public int vocabSize() {
        return vocab_size;
    }

    public ModelTrainer createTrainer() {
        return new ModelTrainer();
    }

    public Model getModel() {
        return new Model();
    }

    public void saveModel(String filepath) throws IOException {
        File file = new File(filepath + "/vocabulary.txt");
        PrintStream fout = new PrintStream(file);
        fout.println(vocab_size);
        for (String word : vocabulary.getEntries()) {
            fout.println(word);
        }
        fout.close();
        file = new File(filepath + "/coocurences.txt");
        fout = new PrintStream(file);
        fout.println(cooccurences.getWindowSize());
        fout.println(String.valueOf(cooccurences.getSymmetric()));
        for (int i = 0; i < vocab_size; i++) {
            StringBuilder str = new StringBuilder();
            for (int j = 0; j < vocab_size; j++) {
                double crc = cooccurences.getValue(i, j);
                if (crc > 0d) {
                    str.append(j);
                    str.append("\t");
                    str.append(cooccurences.getValue(i, j));
                    str.append("\t");
                }
            }
            fout.println(str.toString());
        }
        fout.close();
        model.saveModel(filepath + "/model.txt");
    }

    public void loadModel(String filepath) throws IOException {
        if (model != null || vocabulary != null || cooccurences != null)
            throw new LoadingModelException("You've already started constructing this model. Please, create the new one for loading.");

        File file = new File(filepath + "/vocabulary.txt");
        BufferedReader fin;
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find vocabulary file to load the model from.");
        }
        vocab_size = Integer.parseInt(fin.readLine());
        List<String> words = new ArrayList<>();
        for (int i = 0; i < vocab_size; i++)
            words.add(fin.readLine());
        vocabulary = new Vocabulary(words);
        fin.close();
        file = new File(filepath + "/coocurences.txt");
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find coocurences file to load the model from.");
        }
        int window = Integer.parseInt(fin.readLine());
        boolean symmetry = Boolean.parseBoolean(fin.readLine());
        SparseVec[] crcs = new SparseVec[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            crcs[i] = new SparseVec(vocab_size);
            for (int j = 0; j < vocab_size; j++) {
                crcs[i].set(j, 0d);
            }
            String s = fin.readLine();
            if (s.isEmpty()) continue;
            String[] values = s.split("\t");
            for (int k = 0; k < values.length; k += 2) {
                int j = Integer.parseInt(values[k]);
                crcs[i].set(j, Double.parseDouble(values[k + 1]));
            }
        }
        cooccurences = new Cooccurences(vocab_size, window, symmetry, crcs);
        fin.close();
        file = new File(filepath + "/model.txt");
        try {
            fin = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find model file to load the model from.");
        }
        String modelName = fin.readLine();
        model = ModelChooser.model(modelName, vocabulary, cooccurences);
        fin.close();
        model.loadModel(filepath + "/model.txt");
    }

    public class Model {

        public Model() {
            model.prepareReadyModel();
        }
        public String wordsDifference(String word1, String word2) {
            final ArrayVec v1 = model.getVectorByWord(word1);
            final ArrayVec v2 = model.getVectorByWord(word2);
            String result = "";
            try {
                result = getClosest(ArrayVector.vectorsDifference(v1, v2), "");
            } catch (Word2VecUsageException e) {
                throw new Word2VecUsageException("There is no word with the meaning close to " + word1 + " - " + word2);
            }
            return result;
        }

        public String addWord(String word1, String word2) {
            ArrayVec v1 = model.getVectorByWord(word1);
            ArrayVec v2 = model.getVectorByWord(word2);
            String result = "";
            try {
                result = getClosest(ArrayVector.sumVectors(v1, v2), "");
            } catch (Word2VecUsageException e) {
                throw new Word2VecUsageException("There is no word with the meaning close to " + word1 + " + " + word2);
            }
            return result;
        }

        public String getClosest(String word) {
            return getClosest(model.getVectorByWord(word), word);
        }

        private String getClosest(ArrayVec v1, String word) {
            double minNorm = Double.MAX_VALUE;
            String closest = null;
            for (String word2 : vocabulary.getEntries()) {
                final ArrayVec v2 = model.getVectorByWord(word2);
                final double norm = ArrayVector.countVecNorm(ArrayVector.vectorsDifference(v1, v2));
                if (norm < minNorm && !word.equals(word2)) {
                    minNorm = norm;
                    closest = word2;
                }
            }
            if (closest == null) {
                throw new Word2VecUsageException("There is no word with the meaning close to " + word);
            }
            return closest;
        }
    }

    public class ModelTrainer {
        public void buildVocab(String filepath) throws VocabularyBuildingException {
            vocabulary = new Vocabulary(filepath);
            vocab_size = vocabulary.size();
        }

        private void training(String modelName) {
            if (model == null)
                model = ModelChooser.model(modelName, vocabulary, cooccurences);
            model.trainModel();
        }

        public void trainModel(String filepath) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath);
            training(DEFAULT_MODEL);
        }

        public void trainModel(String filepath, boolean windown_symmetry) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath, windown_symmetry);
            training(DEFAULT_MODEL);
        }

        public void trainModel(String filepath, int window_size) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath, window_size);
            training(DEFAULT_MODEL);
        }

        public void trainModel(String filepath, int window_size, boolean windown_symmetry) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath, window_size, windown_symmetry);
            training(DEFAULT_MODEL);
        }

        public void trainModel(String filepath, String modelName) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath);
            training(modelName);
        }

        public void trainModel(String filepath, boolean windown_symmetry, String modelName) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath, windown_symmetry);
            training(modelName);
        }

        public void trainModel(String filepath, int window_size, String modelName) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath, window_size);
            training(modelName);
        }

        public void trainModel(String filepath, int window_size, boolean windown_symmetry, String modelName) throws CooccurencesBuildingException {
            if (cooccurences == null)
                cooccurences = new Cooccurences(vocabulary, filepath, window_size, windown_symmetry);
            training(modelName);
        }
    }
}
