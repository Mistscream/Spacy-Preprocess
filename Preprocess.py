import spacy
from spacy_iwnlp import spaCyIWNLP

class Preprocess:
    # zur Lemmatisierung im Deutschen
    iwnlp = spaCyIWNLP(lemmatizer_path='data/IWNLP.Lemmatizer_20181001.json')

    nlp = spacy.load('de')

    try:
        # add pipes
        nlp.add_pipe(iwnlp)
        # nlp.add_pipe(__set_custom_boundaries, before='parser')
    except Exception:
        pass

    stopwords_to_remove_from_default_set = ["schlecht", "mensch", "menschen", "beispiel", "gott", "jahr", "jahre",
                                            "jahren", "nicht", "uhr"]
    for stopword in stopwords_to_remove_from_default_set:
        nlp.vocab[stopword].is_stop = False

    #Spacy Token Tags, which will be removed by preprocessing
    tags_to_remove = ['$(', '$,', '$.', 'APPR', 'APPO', 'APPRART', 'APZR', 'ART', 'ITJ', 'KOKOM',
                      'KON', 'KOUI', 'KOUS',  # 'CARD',
                      'PDS', 'PAV', 'PROAV', 'PDAT', 'PIAT', 'PIDAT', 'PIS', 'PPER', 'PPOSAT',
                      'PPOSS', 'PRELAT', 'PRELS', 'PRF', 'PTKA',  # 'PTKANT',
                      'PTKVZ', 'PTKZU', 'PWAT', 'PWAV', 'PWS', 'TRUNC', 'XY', 'SP',
                      'WRP']

    def __init__(self, text, maintain_indeces=None, split_in_sentences=True):
        self.text = text
        self.nlp_text = self.nlp(text)

        if maintain_indeces is None:
            self.maintain_indeces = []
        else:
            self.maintain_indeces = maintain_indeces

        self.noun_chunks = self.get_noun_chunks(cleaned=True, flattened=True)
        self.maintain_indeces.extend(index for index in self.noun_chunks if index not in self.maintain_indeces)

        self.named_entities = self.get_named_entities(flattened=True)
        self.maintain_indeces.extend(index for index in self.named_entities if index not in self.maintain_indeces)
        self.maintain_indeces.sort()

        self.preprocessed = self.preprocess(sentence_split=split_in_sentences)



    def __get_lemma(self, token):
        '''
        take lemma of IWNLP, if given, else spacy lemma
        :param token: spacy-token
        :return: lemmatization
        '''
        lemma_spacy = token.lemma_
        lemma_iwnlp_list = token._.iwnlp_lemmas
        if lemma_iwnlp_list:
            lemma_iwnlp = lemma_iwnlp_list[0]
            return lemma_iwnlp

        return lemma_spacy




    def get_named_entities(self, only_indeces=True, flattened=False):
        '''
        return array of named entities (PER: Person, LOC: Location, ORG: Named corporate, governmental, or other organizational entity, MISC: Miscellaneous entities, e.g. events, nationalities, products or works of art)
        :param only_indeces:
        :param flattened: returns only 1d array, else related entities are in sup-arrays
        :return: array with named entities
        '''
        if flattened:
            named_ents = [word.i if only_indeces else (word.i, word, ents.label_) for ents in self.nlp_text.ents for word in ents]
        else:
            named_ents = [[word.i if only_indeces else (word.i, word, ents.label_) for word in ents] for ents in self.nlp_text.ents]
        return named_ents

    def get_noun_chunks(self, only_indices=True, cleaned=True, flattened=False):
        '''
        return array of noun_chunks/noun_phrases of the Text object
        :param only_indices:
        :param cleaned: noun phrases without stopword, punctuation
        :param flattened: returns only 1d array, else related phrases are in sup-arrays
        :return: array with noun-phrases
        '''

        # noun_words = [(word.i, word) for ent in text.noun_chunks for word in ent]
        # noun_words = [[(word.i, word) for word in ent] for ent in text.noun_chunks]
        if flattened:
            if cleaned:
                noun_words = [word.i if only_indices else (word.i, word)
                              for ent in self.nlp_text.noun_chunks
                              for word in ent
                              if self.__is_valid_token(word)]
            else:
                noun_words = [word.i if only_indices else (word.i, word)
                              for ent in self.nlp_text.noun_chunks
                              for word in ent]
        else:
            if cleaned:
                noun_words = [[word.i if only_indices else (word.i, word) for word in ent
                               if self.__is_valid_token(word)]
                              for ent in self.nlp_text.noun_chunks]
            else:
                noun_words = [[word.i if only_indices else (word.i, word) for word in ent]
                              for ent in self.nlp_text.noun_chunks]

        return noun_words

    def __is_valid_token(self, token):
        '''
        checks if token is valid: no stopword, punctuation oder whitespace
        :param token: spacy-token
        :return: bool
        '''
        # nlp(token.lower_)[0] wegen spacy bug --> z.B. "Der" würde nicht als stopwort erkannt werden, "der" aber schon
        if not self.nlp(token.lower_)[0].is_stop and not token.is_punct and not token.is_space:
            return True

        return False

    def __tokenize_words(self, doc):
        '''
        tokenizes text and removes unimportant tokens
        :return: 1d array of tokens
        '''
        tokenized_text = [self.__get_lemma(token).lower() for token in doc
                             if self.__is_valid_token(token)
                             and not token.tag_ in self.tags_to_remove
                             or token.i in self.maintain_indeces]
        return tokenized_text

    def __tokenize_to_list_sentences(self):
        '''
        tokenizes text and removes unimportant tokens, split by sentences
        :return: 2d array of tokens in sub-arrays (sentences)
        '''
        filtered_text = []

        for sentence in self.nlp_text.sents:
            filtered_sentence = self.__tokenize_words(sentence)
            filtered_text.append(filtered_sentence)

        return filtered_text

    def preprocess(self, sentence_split=True):
        '''
        preprocess text. removes unimportant tokens
        :param sentence_split: split by sentences
        :return: preprocessed text as 1d or 2d-array (sentences)
        '''
        if sentence_split:
            preprocessed_text = self.__tokenize_to_list_sentences()
        else:
            preprocessed_text = self.__tokenize_words(self.nlp_text)

        return preprocessed_text


