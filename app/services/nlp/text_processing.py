import spacy
import re

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str):
    """
    Clean and normalize merged text.
    Output:
      - clean_text (full cleaned string)
      - sentences (list of cleaned sentences)
    """

    if not text:
        return "", []

    # 1. Basic cleanup
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # remove multiple spaces
    text = text.replace("\n", " ")

    # 2. Lowercase for normalization
    text = text.lower()

    # 3. spaCy NLP processing
    doc = nlp(text)

    sentences = []
    cleaned_tokens = []

    for sent in doc.sents:
        # Clean each sentence
        clean_sentence = []

        for token in sent:
            # Skip punctuation, spaces, stopwords
            if token.is_stop or token.is_punct or token.is_space:
                continue

            # Lemmatize the token
            lemma = token.lemma_

            # Remove special characters
            lemma = re.sub(r"[^a-zA-Z0-9]+", "", lemma)

            if lemma:
                clean_sentence.append(lemma)
                cleaned_tokens.append(lemma)

        # Convert tokens back into cleaned sentence
        if clean_sentence:
            sentences.append(" ".join(clean_sentence))

    # Combine all cleaned tokens for bias model
    clean_text = " ".join(cleaned_tokens)

    return clean_text, sentences
