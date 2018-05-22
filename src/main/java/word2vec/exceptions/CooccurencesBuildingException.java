package word2vec.exceptions;

import word2vec.text_utils.Cooccurences;

public class CooccurencesBuildingException extends RuntimeException {
    public CooccurencesBuildingException() {super();}
    public CooccurencesBuildingException(String message) { super(message); }
}
