import spacy
from spacy_iwnlp import spaCyIWNLP

from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

from spacy.lang.char_classes import ALPHA

from iwnlp.iwnlp_wrapper import IWNLPWrapper

def custom_tokenizer(nlp):
    infixes = list(nlp.Defaults.infixes)

    # add custom tokenize cases:
    # for case: <Wort>-<Wort> --> für deutsch eher weglassen?
    #infixes.append(r'(?<=[{a}"])[-](?=[{a}])'.format(a=ALPHA))
    # for case: <Zahl>-<Wort>
    infixes.append(r'(?<=[0-9])[-](?=[{a}])'.format(a=ALPHA))

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab,
                     rules=nlp.Defaults.tokenizer_exceptions,
                     prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None)


class Preprocess:
    # zur Lemmatisierung im Deutschen


    nlp = spacy.load('de')

    # IWNLP German Lemmatizations:
    #iwnlp = spaCyIWNLP(lemmatizer_path='data/IWNLP.Lemmatizer_20181001.json', ignore_case=True)
    lemmatizer = IWNLPWrapper(lemmatizer_path='data/IWNLP.Lemmatizer_20181001.json')

    #add custom tokenizer
    nlp.tokenizer = custom_tokenizer(nlp)

    '''
    try:
        # add pipes
        nlp.add_pipe(iwnlp)
        # nlp.add_pipe(__set_custom_boundaries, before='parser')
    except Exception:
        pass
    '''

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


    def __init__(self, text, split_in_sentences=True, with_pos=False):
        '''

        :param text: input text
        :param split_in_sentences: split text in sentences --> sub-arrays for sentences in Preprocess-result
        :param with_pos: true: give tripel with (<startpos in orig-text>, <endpos in origtext>, token), else only tokens
        '''
        self.text = text
        self.nlp_text = self.nlp(text)

        self.maintain_indeces = []

        self.noun_chunks = self.get_noun_chunks(cleaned=True, flattened=True)
        self.maintain_indeces.extend(index for index in self.noun_chunks if index not in self.maintain_indeces)

        self.named_entities = self.get_named_entities(flattened=True)
        self.maintain_indeces.extend(index for index in self.named_entities if index not in self.maintain_indeces)
        self.maintain_indeces.sort()

        self.preprocessed = self.preprocess(sentence_split=split_in_sentences, with_pos=with_pos)



    def __get_lemma(self, token):
        '''
        take lemma of IWNLP, if given, else spacy lemma
        :param token: spacy-token
        :return: lemmatization
        '''
        #lemma_iwnlp_list = token._.iwnlp_lemmas
        lemma_iwnlp_list = self.lemmatizer.lemmatize_plain(token.text, ignore_case=False)
        if lemma_iwnlp_list:
            lemma_iwnlp = lemma_iwnlp_list[0]
            #print(token, ":::", lemma_iwnlp_list[0])
            return lemma_iwnlp

        return token.lemma_




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

    def __tokenize_words(self, doc, with_pos=False):
        '''
        tokenizes text and removes unimportant tokens
        :param doc: input spacy doc
        :param with_pos: true: give tripel with (<startpos in orig-text>, <endpos in origtext>, token), else only tokens
        :return: 1d array of tokens
        '''
        tokenized_text = [(token.idx, token.idx + len(token), self.__get_lemma(token).lower()) if with_pos else self.__get_lemma(token).lower() for token in doc
                          if self.__is_valid_token(token)
                          and not token.tag_ in self.tags_to_remove
                          or token.i in self.maintain_indeces]

        return tokenized_text


    def __tokenize_to_list_sentences(self, with_pos=False):
        '''
        tokenizes text and removes unimportant tokens, split by sentences
        :param with_pos: true: give tripel with (<startpos in orig-text>, <endpos in origtext>, token), else only tokens
        :return: 2d array of tokens in sub-arrays (sentences)
        '''
        filtered_text = []
        for sentence in self.nlp_text.sents:
            filtered_sentence = self.__tokenize_words(sentence, with_pos=with_pos)
            filtered_text.append(filtered_sentence)

        return filtered_text

    def preprocess(self, sentence_split=True, with_pos=False):
        '''
        preprocess text. removes unimportant tokens
        :param sentence_split: split by sentences
        :param with_pos: true: give tripel with (<startpos in orig-text>, <endpos in origtext>, token), else only tokens
        :return: 1d or 2d array with preprocessed text
        '''
        if sentence_split:
            preprocessed_text = self.__tokenize_to_list_sentences(with_pos=with_pos)
        else:
            preprocessed_text = self.__tokenize_words(self.nlp_text, with_pos=with_pos)

        return preprocessed_text


