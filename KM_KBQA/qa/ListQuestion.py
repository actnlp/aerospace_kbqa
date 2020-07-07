from ..common import LTP
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
        rel_word = ['时间', '几点', '地址', '地方', '几楼',
                    '怎么走', '位置', '电话', '联系', '价格', '费', '钱']
        return any(map(lambda word: word in sent, rel_word))

    has_rel = has_rel_word(sent)
    if has_rel:
        return False
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
            if word_loc >= 0 and word_loc < min_start and not has_rel:
                return True
    return False


def generate_ngram(seq, max_size=6, min_size=1, ignore_w=[]):
    for n in range(min_size, max_size+1):
        for j in range(len(seq)-n+1):
            # delete word consists of two single token or in ignore_w
            cur_w = ''.join(seq[j:j+n])
            if len(cur_w) <= 1 and cur_w != '有':
                continue
            if cur_w == '服务':
                continue
            if len(list(filter(lambda x: x in cur_w, ignore_w))) > 0:
                continue
            if ''.join(seq[j:j+n]) in ignore_w:
                continue
            if n == 2 and len(seq[j]) == 1 and len(seq[j+1]) == 1 and seq[j] != '几':
                continue
            yield (''.join(seq[j:j+n]), j)


def retrieve_ngram(sent, ignore_w=[]):
    return list(generate_ngram(LTP.cut(sent), 3, 1, ignore_w))


# class WordCLS:
#     def __init__(self, sent, bc, thres, conf_lw, ignore_w=[]):
#         self.sent = sent
#         self.bc = bc
#         # self.conf_lw = ['几个', '哪些', '多少', '什么']
#         self.conf_lw = conf_lw # ['银行', '货币兑换', '快递', '出行', '餐饮', '便利店', '充电', '便民']
#         self.ignore_w = ignore_w
#         self.cut_words = retrieve_ngram(sent, ignore_w)
#         self.conf_lw_embed = [self.bc.encode([word]) for word in self.conf_lw]
#         self.thres = thres
#
#     def filter_similar_words(self, cut_words):
#         # calculate all possible word embedding
#         # print(cut_words)
#         cut_words = [word[0] for word in cut_words if len(word[0]) > 1 or word[0] == '有']
#         if len(cut_words) == 0:
#             return [], None
#         # for word in cut_words:
#         #     if word in self.conf_lw:
#         #         confirmed_list_words.append(word)
#         cut_words_emb = [self.bc.encode([word]) for word in cut_words]
#
#         # iter through two sets, compare with threshold
#         selected_list_words = dict()
#         min_i, min_j, min_d = 0, 0, 1000
#         for i, conf in enumerate(self.conf_lw_embed):
#             for j, cand in enumerate(cut_words_emb):
#                 dist = np.sum((cand - conf) ** 2).item()
#                 simi = torch.cosine_similarity(torch.tensor(cand), torch.tensor(conf)).item()
#                 # if there is intersection between cand and conf
#                 if cut_words[j] in self.conf_lw[i] or self.conf_lw[i] in cut_words[j]:
#                     dist /= 1.5
#                     simi = min(0.99, simi * 1.1)
#                 if dist < min_d and simi > 0.831:
#                     min_i, min_j, min_d = i, j, dist
#                 if dist < self.thres:
#                     # if cut_words[j] not in selected_list_words.keys():
#                     selected_list_words[(cut_words[j], self.conf_lw[i])] = dist
#                     # else:
#                         # selected_list_words[cut_words[j]] = min(dist, selected_list_words[cut_words[j]])
#
#         # print(selected_list_words)
#         min_dist_word = [cut_words[min_j], self.conf_lw[min_i], min_d]
#         return selected_list_words, min_dist_word
#
#     def classify_word(self):
#         return self.filter_similar_words(self.cut_words)[1]
#
#     def has_similar_words(self, asserted_w=[]):
#         flt = list(filter(lambda x: x in self.sent, asserted_w))
#         if len(flt) > 0:
#             return False
#         return len(self.filter_similar_words(self.cut_words)[0]) > 0

if __name__ == '__main__':
    pass
