package word2vec;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.SparseMx;
import com.expleague.commons.util.ArrayTools;
import word2vec.exceptions.*;
import word2vec.models.AbstractModelFunction;
import word2vec.models.ModelChooser;
import word2vec.text_utils.CooccurencesBuilder;
import word2vec.text_utils.Vocabulary;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Word2Vec {

    private Vocabulary vocabulary;
    private int vocab_size;
    private Mx cooccurences;
    private int leftWindow;
    private int rightWindow;

    private AbstractModelFunction model;

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
        fout.println(leftWindow);
        fout.println(rightWindow);
        for (int i = 0; i < vocab_size; i++) {
            StringBuilder str = new StringBuilder();
            for (int j = 0; j < vocab_size; j++) {
                double crc = cooccurences.get(i, j);
                if (crc > 0d) {
                    str.append(j);
                    str.append("\t");
                    str.append(cooccurences.get(i, j));
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

        try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath + "/vocabulary.txt")))){
            vocab_size = Integer.parseInt(fin.readLine());
            List<String> words = new ArrayList<>();
            for (int i = 0; i < vocab_size; i++)
                words.add(fin.readLine());
            vocabulary = new Vocabulary(words);
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find vocabulary file to load the model from.");
        }
        try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath + "/coocurences.txt")))) {
            leftWindow = Integer.parseInt(fin.readLine());
            rightWindow = Integer.parseInt(fin.readLine());
            Mx crcs = new SparseMx(vocab_size, vocab_size);
            for (int i = 0; i < vocab_size; i++) {
                String s = fin.readLine();
                if (s.isEmpty()) continue;
                String[] values = s.split("\t");
                for (int k = 0; k < values.length; k += 2) {
                    int j = Integer.parseInt(values[k]);
                    crcs.set(i, j, Double.parseDouble(values[k + 1]));
                }
            }
            cooccurences = crcs;
        }
        try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath + "/model.txt")))) {
            String modelName = fin.readLine();
            model = ModelChooser.model(modelName, vocabulary, cooccurences);
            fin.close();
            model.loadModel(filepath + "/model.txt");
        }
    }

    public class Model {
        public Model() {
            model.prepareReadyModel();
        }

        public double countLikelihood() {
            return model.likelihood();
        }

        public boolean isCloser(String objectWord, String closerWord, String furtherWord) {
            return model.getDistance(objectWord, closerWord) < model.getDistance(objectWord, furtherWord);
        }

        public List<String> wordsDifference(String word1, String word2) {
            final Vec v1 = VecTools.copy(model.getVectorByWord(word1));
            final Vec v2 = model.getVectorByWord(word2);
            VecTools.incscale(v1, v2, -1);
            List<String> result;
            try {
                result = getClosest(v1, "");
            } catch (Word2VecUsageException e) {
                throw new Word2VecUsageException("There is no word with the meaning close to " + word1 + " - " + word2);
            }
            return result;
        }

        public void sdf() {
            int[] order = ArrayTools.sequence(0, vocab_size);
            double[] weights = IntStream.of(order).mapToDouble(idx -> {
                return model.getSkewVector(vocab().get(idx));
            }).toArray();
            ArrayTools.parallelSort(weights, order);
            IntStream.of(order).forEach(idx -> {
                String word = vocab().get(idx);
                System.out.println(word + "\t" + model.getSkewVector(word));
            });
        }

        public List<String> addWord(String word1, String word2) {
            Vec v1 = model.getVectorByWord(word1);
            Vec v2 = model.getVectorByWord(word2);
            List<String> result;
            try {
                result = getClosest(VecTools.sum(v1, v2), "");
            } catch (Word2VecUsageException e) {
                throw new Word2VecUsageException("There is no word with the meaning close to " + word1 + " + " + word2);
            }
            return result;
        }

        public List<String> getClosest(String word) {
            return getClosest(model.getVectorByWord(word), word);
        }

        private List<String> getClosest(Vec v1, String word) {
            return model.getWordByVector(v1);
            /*double minNorm = Double.MAX_VALUE;
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
            return closest;*/
        }
    }

    public class ModelTrainer {
        public void buildVocab(String filepath) throws VocabularyBuildingException {
            vocabulary = new Vocabulary(filepath);
            vocab_size = vocabulary.size();
        }

        public void trainModel(ModelParameters modelParameters) throws CooccurencesBuildingException {
            leftWindow = modelParameters.getLeftWindow();
            rightWindow = modelParameters.getRightWindow();

            if (cooccurences == null) {
                try (final BufferedReader bufferedReader = Files.newBufferedReader(Paths.get(modelParameters.getFilepath()))) {
                    cooccurences = new CooccurencesBuilder()
                        .setLeftWindow(leftWindow)
                        .setRightWindow(rightWindow)
                        .setVocabulary(vocabulary)
                        .build(bufferedReader);
                }
                catch (IOException e) {
                    throw new RuntimeException(e);
                }
                if (model == null)
                    model = ModelChooser.model(modelParameters.getModelName(), vocabulary, cooccurences);
                model.trainModel();
            }
        }
    }
}
