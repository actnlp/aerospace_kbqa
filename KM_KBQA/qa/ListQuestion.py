from ..config import config


def check_list_questions(sent, link_res):
    # def has_rel_word(word, prop):
    #     ret = False
    #     if '时间' in prop and '时间' in word or '时间' in prop and '几点' in word:
    #         ret = True
    #     if prop == word or '地址' in word and '地点' in prop or '地点' in prop and '地方' in word or '地点' in prop and '几楼' in word or '怎么走' in word and '地点' in prop or '位置' in word and '地点' in prop:
    #         ret = True
    #     if '联系' in prop and '电话' in word or '电话' in prop and '联系' in word or word == prop:
    #         ret = True
    #     money = ['价格', '费', '钱']
    #     for p_i in money:
    #         for w_i in money:
    #             if p_i in prop and w_i in word:
    #                 ret = True
    #     return ret
    def has_rel_word(sent):
        rel_word = ['有', '时间', '几点', '地址', '地方', '几楼',
                    '怎么走', '位置', '电话', '联系', '价格', '费', '钱']
        return any(map(lambda word: word in sent, rel_word))

    #has_rel = has_rel_word(sent)
    #if has_rel:
    #    return False
    # for mention in mention_list:
    #     # if has_rel_word(mention, '地点') or has_rel_word(mention, '时间') or has_rel_word(mention, '电话') or has_rel_word(mention, '价格'):
    #     #     has_rel = True
    #     has_rel = has_rel \
    #         or has_rel_word(mention, '地点') \
    #         or has_rel_word(mention, '时间') \
    #         or has_rel_word(mention, '电话') \
    #         or has_rel_word(mention, '价格')

    if len(link_res) > 0:
        min_start = min(map(lambda ent: sent.find(ent.get('mention', '')),
                            link_res))
        # 查找是否有一个listword在句子里面
        for list_word in config.LIST_WORD_LIST:
            word_loc = sent.find(list_word)
            if word_loc >= 0 and word_loc < min_start:
                return True
    return False
