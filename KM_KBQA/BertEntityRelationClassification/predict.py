# coding: utf-8
# predict methods for different entities
import pdb
import jieba
import logging

logger = logging.getLogger()


def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / len(a | b)


def jaccard_rank(entities):
    pass


CONTENT_MATCH_THRESH = 0.5


def merchandise_predict(sentence, driver):
    # cut sentence
    words = jieba.lcut(sentence)
    logger.info('cut '+str(words))
    results = driver.get_merchandise_entities_by_genre('SubGenre').result()

    merchandises = [[res[0]['name'], res[0]
                     ['服务内容'].split(';')] for res in results]

    match_entities = {}
    for word in words:
        if len(word) <= 1:
            continue
        for mer in merchandises:
            match = map(lambda m: jaccard(word, m), mer[1])
            max_score = max(match)
            if max_score > CONTENT_MATCH_THRESH:
                if mer[0] in match_entities:
                    max_score = max(match_entities[mer[0]][1], max_score)
                match_entities[mer[0]] = (mer[0], max_score)
    match_entities = list(match_entities.values())
    match_entities.sort(key=lambda x: x[1], reverse=True)
    match_entities = [m[0] for m in match_entities[:3]]
    return match_entities


def match_content(sent_cut: list, contents: list):
    ''''
    Args:
        sent_cut: ['word', 'word']
        contents: [
            [entity_name:str, entity_content:list], 
        ]
    '''
    words_set = set(sent_cut)
