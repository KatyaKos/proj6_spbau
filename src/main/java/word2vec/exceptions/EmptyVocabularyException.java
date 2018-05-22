package word2vec.exceptions;

public class EmptyVocabularyException extends RuntimeException{
    public EmptyVocabularyException() {super();}
    public EmptyVocabularyException(String message) {super(message);}
}
