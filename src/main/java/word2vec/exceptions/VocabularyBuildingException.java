package word2vec.exceptions;

public class VocabularyBuildingException extends RuntimeException{
    public VocabularyBuildingException() { super(); }
    public VocabularyBuildingException(String message) { super(message); }
}
