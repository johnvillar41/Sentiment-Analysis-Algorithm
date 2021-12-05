from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.util import ngrams

from Models.SentiwordModel import SentiwordModel

lemmatizer = WordNetLemmatizer()


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def clean_text(text):
    text = text.replace("<br />", " ")
    return text


def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """

    polarityScore = 0.0
    tokens_count = 0
    sentimentScore = ""
    text = clean_text(text)

    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:

        unigramSent = 0.0
        bigramSent = 0.0

        ##############################
        #       Bigram Model
        ##############################

        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        tokens, tags = zip(*tagged_sentence)
        bigramTokens = []
        bigramTags = []
        tokenBigrams = ngrams(tokens, 2)
        tagBigrams = ngrams(tags, 2)

        for t1, t2 in tokenBigrams:
            bigramTokens.append((t1, t2))

        for t1, t2 in tagBigrams:
            bigramTags.append((t1, t2))

        for i in range(0, len(bigramTokens)):
            wn_tag1 = penn_to_wn(bigramTags[i][0])
            if wn_tag1 not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            wn_tag2 = penn_to_wn(bigramTags[i][1])
            if wn_tag2 not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma1 = lemmatizer.lemmatize(bigramTokens[i][0], pos=wn_tag1)
            if not lemma1:
                continue

            lemma2 = lemmatizer.lemmatize(bigramTokens[i][1], pos=wn_tag2)
            if not lemma2:
                continue

            synsets1 = wn.synsets(lemma1, pos=wn_tag1)
            if not synsets1:
                continue

            synsets2 = wn.synsets(lemma2, pos=wn_tag2)
            if not synsets2:
                continue

            # Take the first sense, the most common
            synset1 = synsets1[0]
            synset2 = synsets2[0]
            swn_synset1 = swn.senti_synset(synset1.name())
            swn_synset2 = swn.senti_synset(synset2.name())

            tmp1sent = swn_synset1.pos_score() - swn_synset1.neg_score()
            tmp2sent = swn_synset2.pos_score() - swn_synset2.neg_score()

            if tmp1sent < 0 and tmp2sent < 0:
                bigramSent += -1 * min(tmp1sent, tmp2sent)
            elif tmp1sent > 0 and tmp2sent > 0:
                bigramSent += max(tmp1sent, tmp2sent)
            else:
                bigramSent += min(tmp1sent, tmp2sent)

        ###########################
        #      Unigram Model
        ###########################

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())            
        
            unigramSent += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

        if max(abs(bigramSent), abs(unigramSent)) == abs(bigramSent):
            polarityScore += bigramSent
        else:
            polarityScore += unigramSent

        if polarityScore > 1:
            polarityScore = 1
        elif polarityScore < -1:
            polarityScore = -1

        if polarityScore > 0:
            sentimentScore = "Positive"
        elif polarityScore < 0:
            sentimentScore = "Negative"
        else:
            sentimentScore = "Neutral"


        sentiword = SentiwordModel(polarityScore, sentimentScore)
    return sentiword.toJSON()
