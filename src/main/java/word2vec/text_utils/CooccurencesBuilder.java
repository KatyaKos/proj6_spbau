package word2vec.text_utils;


import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.SparseMx;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.util.logging.Interval;
import gnu.trove.list.array.TIntArrayList;

import java.io.Reader;
import java.text.BreakIterator;

// Table X = {Xij}, Xij = number of times word j occures in the context of the word i.
// If not symmetric - only looks at word's left neighbours.
public class CooccurencesBuilder {
    private int leftWindow;
    private int rightWindow;
    private Vocabulary voc;

    public CooccurencesBuilder setVocabulary(Vocabulary voc) {
        this.voc = voc;
        return this;
    }

    public CooccurencesBuilder setLeftWindow(int leftWindow) {
        this.leftWindow = leftWindow;
        return this;
    }

    public CooccurencesBuilder setRightWindow(int rightWindow) {
        this.rightWindow = rightWindow;
        return this;
    }

    public Mx build(Reader reader) throws RuntimeException {
        final int vocSize = this.voc.size();
        final Mx result = new SparseMx(vocSize, vocSize);

        Interval.start();
        CharSeqTools.lines(reader).parallel().forEach(line -> {
            BreakIterator breakIterator = BreakIterator.getWordInstance();
            breakIterator.setText(line.toString());
            int lastIndex = breakIterator.first();
            final TIntArrayList queue = new TIntArrayList(1000);

            while (BreakIterator.DONE != lastIndex) {
                int firstIndex = lastIndex;
                lastIndex = breakIterator.next();
                if (lastIndex != BreakIterator.DONE && Character.isLetterOrDigit(line.charAt(firstIndex))) {
                    final CharSeq word = line.sub(firstIndex, lastIndex);
                    int wordId = voc.wordToIndex(word.toString());
                    if (wordId == Vocabulary.NO_ENTRY_VALUE) {
                        continue;
                    }
                    queue.add(wordId);
                }
            }
            final SparseMx temp = new SparseMx(vocSize, vocSize);
            System.out.print("Colocation harvesting: passed 0 items");
            for (int i = 0; i < queue.size(); i++) {
                final int indexedId = queue.get(i);
                final int rightLimit = Math.min(queue.size(), i + rightWindow + 1);
                final int leftLimit = Math.max(0, i - leftWindow);
                for (int idx = leftLimit; idx < rightLimit; idx++) {
                    if (idx == 0)
                        continue;
                    temp.adjust(indexedId, queue.get(idx), 1);
                }
                if ((i + 1) % 1000000 == 0) {
                    System.out.print("\rColocation harvesting: passed " + (i + 1) + " items");
                    synchronized (result) {
                        VecTools.append(result, temp);
                        temp.clear();
                    }
                }
            }
            System.out.println();
            synchronized (result) {
                VecTools.append(result, temp);
            }
        });
        Interval.stopAndPrint("Cooccurrences calculated for: ");
        return result;
    }
}