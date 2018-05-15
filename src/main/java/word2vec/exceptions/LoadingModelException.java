package word2vec.exceptions;

public class LoadingModelException extends RuntimeException{
    public LoadingModelException() { super(); }
    public LoadingModelException(String message) { super(message); }

}
