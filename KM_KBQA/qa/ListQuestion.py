from ..config import config
import pandas as pd


def check_list_questions(sent):
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

    # 是否是可以回答的列举类问题
    def list_able_content(sent):
        flag = False
        for word in config.LIST_CONTENT_LIST:
            if word in sent:
                flag = True
                break
        return flag

    if list_able_content(sent):
        # min_start = min(map(lambda ent: sent.find(ent.get('mention', '')),
        #                     link_res))
        # 查找是否有一个listword在句子里面
        lexicon = pd.read_csv(config.AEROSPACE_LEXICON_PATH, sep="\n", header=None)[0]
        num_property = [w.split(" ")[0] for w in lexicon if "数量" in w and w != '数量']
        for list_word in config.LIST_WORD_LIST:
            if list_word == '有' and any([word in sent for word in ['有限','有多重','有多高']]):
                continue
            if list_word == '数量' and any([word in sent for word in num_property]):
                continue
            if any([word in sent for word in ['范围','分类','部分','结构','工作人员','职位','服务','缺点','优点','用处','作用']]):
                continue
            word_loc = sent.find(list_word)
            if word_loc >= 0:
                return True
        if sent in ['国内航空公司','国内廉价航空公司','廉价航空公司','中国航空公司','国内机场','中国机场']:
            return True
    return False
