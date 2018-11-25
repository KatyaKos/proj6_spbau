package com.expleague.ml.embedding;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.SparseMx;
import com.expleague.ml.embedding.exceptions.*;
import com.expleague.ml.embedding.model_functions.AbstractModelFunction;
import com.expleague.ml.embedding.model_functions.ModelChooser;
import com.expleague.ml.embedding.text_utils.CooccurencesBuilder;
import com.expleague.ml.embedding.text_utils.Vocabulary;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Word2Vec {

    private Vocabulary vocabulary;
    private int vocab_size;
    private Mx cooccurences;
    private int leftWindow;
    private int rightWindow;

    private AbstractModelFunction model;

    public int vocabSize() {
        return vocab_size;
    }

    public ModelTrainer createTrainer() {
        return new ModelTrainer();
    }

    public Model getModel() {
        return new Model(model, vocabulary);
    }

    public void saveModel(String filepath) throws IOException {
        File file = new File(filepath + "/vocab.txt");
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
            for (int j = 0; j < vocab_size; j++) {
                double crc = cooccurences.get(i, j);
                if (crc > 0d) {
                    fout.print(j);
                    fout.print("\t");
                    fout.print(cooccurences.get(i, j));
                    fout.print("\t");
                }
            }
            fout.println();
        }
        fout.close();
        model.saveModel(filepath);
    }

    /**
     * Mode 0 = load vocab + cooccurences + all vectors
     * Mode 1 = load vocab + vectors for evaluation
     */
    public void loadModel(String filepath, int mode) throws IOException {
        if (model != null || vocabulary != null || cooccurences != null)
            throw new LoadingModelException("You've already started constructing this model. Please, create the new one for loading.");

        System.out.println("Loading vocabulary.");
        try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath + "/vocab.txt")))){
            vocab_size = Integer.parseInt(fin.readLine());
            List<String> words = new ArrayList<>();
            for (int i = 0; i < vocab_size; i++)
                words.add(fin.readLine());
            vocabulary = new Vocabulary(words);
        } catch (FileNotFoundException e) {
            throw new LoadingModelException("Couldn't find vocabulary file to load the model from.");
        }
        System.out.println("Vocabulary loaded.");

        if (mode == 0) {
            System.out.println("Loading cooccurences.");
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
            System.out.println("Cooccurences loaded.");
        }

        System.out.println("Loading vectors.");
        try (BufferedReader fin = new BufferedReader(new FileReader(new File(filepath + "/vectors.txt")))) {
            String modelName = fin.readLine();
            model = ModelChooser.model(modelName, vocabulary, cooccurences);
            fin.close();
            model.loadModel(filepath, mode);
        }
        System.out.println("Vectors loaded.");
    }

    public class ModelTrainer {
        public void buildVocab(String filepath) throws VocabularyBuildingException {
            vocabulary = new Vocabulary(filepath);
            vocab_size = vocabulary.size();
        }

        public void trainModel(ModelParameters modelParameters) throws CooccurencesBuildingException {
            if (cooccurences == null) {
                leftWindow = modelParameters.getLeftWindow();
                rightWindow = modelParameters.getRightWindow();
                try (final BufferedReader bufferedReader = Files.newBufferedReader(Paths.get(modelParameters.getFilepath()), StandardCharsets.ISO_8859_1)) {
                    cooccurences = new CooccurencesBuilder()
                        .setLeftWindow(leftWindow)
                        .setRightWindow(rightWindow)
                        .setVocabulary(vocabulary)
                        .build(bufferedReader);
                }
                catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            if (model == null)
                model = ModelChooser.model(modelParameters.getModelName(), vocabulary, cooccurences);
            model.trainModel();
        }
    }
}
