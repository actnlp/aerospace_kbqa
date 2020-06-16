# coding=utf-8
import os 
import pdb
import pickle
import re


class Limiter:
    def __init__(self, sent):
        digits = [('一', '1'), ('二','2'), ('三', '3'),('四', '4'),('五', '5'),('六', '6'),('七', '7'),('八', '8'),('九', '9')]
        if '十' in sent:
            pre_index = sent.index('十') - 1
            if pre_index >= 0 and sent[pre_index] in ''.join([x[0] for x in digits]):
                digits.append(('十', '0'))
        for digit in digits:
            sent = sent.replace(digit[0], digit[1])
        self.sent = sent

    def check_time(self, sent):
        # pattern = re.compile(u'(((早上)|(凌晨)|(上午)|(晚上)|(下午))?(\d)(\d)?(点|时)(\D)?(前|后)?)')
        pattern = re.compile(u'((((\u65E9\u4E0A)|(\u51CC\u6668)|(\u4E0A\u5348)|(\u665A\u4E0A)|(\u4E0B\u5348))?(\d)(\d)?(\u70B9|\u65F6)(\D)?(\u524D|\u540E)?))')
        ret = []
        t_ret = pattern.findall(sent)
        if t_ret is None:
            return ret
        ret_list = list(map(lambda x: x[0], t_ret))
        
        for t in ret_list:
            num_list = list(filter(str.isdigit, t))
            num = int(''.join(num_list))
            if ('下午' in t or '晚上' in t) and num < 12:
                num += 12
            ret.append(str(num))
            '''if '前' in t_ret:
                ret.append(str(num) + '-')
            elif '后' in t_ret:
                ret.append(str(num) + '+')'''

        if '24' in sent or '通宵' in sent or '过夜' in sent:
            ret.append('24小时')

        if '最早' in sent:
            ret.append('最早')

        if '最晚' in sent:
            ret.append('最晚')

        return ret

    def check_location(self, sent):
        # pattern = re.compile(u'(((T|t)\d)|(\d号))?航站楼')
        pattern = re.compile(u'((([Tt]\d)|(\d\u53F7))?\u822A\u7AD9\u697C)|(\d\u697C)|(\d\u5c42)|([Ff]\d)|(\d[Ff])|([Bb]\d)|(\u98de\u673a\u4e0a)')
        rets = pattern.findall(sent)
        for i, ret in enumerate(rets):
            ret = sorted(list(ret), key=lambda x: len(x), reverse=True)
            rets[i] = ret
        if rets is None:
            return rets
        rets = [x[0] for x in rets]
        for ret in rets:
            if '号' in ret:
                ret = ret.replace('号', '')
                ret = 't' + ret

        return rets


    def check_currency(self, sent):
        currencies = ['日元', '港币', '欧元', '美元','英镑','加拿大元','加元','马币', '马来西亚币', '瑞士法郎','澳大利亚元','新西兰纽币','新加坡币','新加坡新币','韩元','印尼盾','泰铢','菲律宾比索','台币','台币','俄罗斯卢布','澳门元','阿联酋迪拉姆','土耳其里拉','印度卢比','印尼卢比','巴林第纳尔','巴西雷亚尔','丹麦克朗','埃及镑', '老挝基普', '斯里兰卡卢比', '缅甸元', '蒙古图格里克', '墨西哥比索', '林吉特、', '挪威克朗', '巴基斯坦卢比', '瑞典克朗', '南非兰特']
        ret = []
        for currency in currencies:
            if currency in sent:
                ret.append(currency)

        return ret

    def check_bank(self, sent):
        banks = ['交通银行','招商银行','农业银行','花旗银行','建设银行','中国银行','工商银行','上海银行','光大银行','北京银行','汇丰银行','工行','建行','农行', '方汇银行','广发银行','恒生联合银行', '艾西益银行', '花旗银行', '交行', '兴业银行'] 
        ret = []
        for bank in banks:
            if bank in sent:
                ret.append(bank)

        return ret

    def check_airline(self, sent):
        airlines = ['南航','海南航空','上海航空','重庆航空','深圳航空','西部航空','东方航空','金鹿航空','新华航空','光大银行','国际航空','南方航空', '国航', '海航', '奥凯航空', '祥鹏','祥鹏航空', '昆明航空', '泰国航空', '山东航空', '东航', '川航', '西藏航空', '春秋航空', '美国航空', '红土航空', '成都航空'
                    ,'福州航空', '长龙航空', '老挝航空', '立荣航空', '瑞丽航空', '亚洲航空', '四川航空', '吉祥航空', '厦门航空', '东海航空', '大韩航空', '联合航空', '首都航空', '天津航空', '百事特']
        ret = []
        for airline in airlines:
            if airline in sent:
                ret.append(airline)

        return ret

    def check_price(self, sent):
        prices = ['免费']
        ret = []
        for price in prices:
            if price in sent:
                ret.append(price)
        return ret

    def check(self):
        limits = {'时间': '', '地点': '', '币种': '', '银行': '', '航空公司': '', '价格': ''}
        time_check = self.check_time(self.sent)
        loc_check = self.check_location(self.sent)
        cur_check = self.check_currency(self.sent)
        bank_check = self.check_bank(self.sent)
        price_check = self.check_price(self.sent)
        airline_check = self.check_airline(self.sent)
        if time_check is not None and len(time_check) > 0:
            limits['时间'] = time_check
        if loc_check is not None and len(loc_check) > 0:
            limits['地点'] = loc_check
        if cur_check is not None and len(cur_check) > 0:
            limits['币种'] = cur_check
        if bank_check is not None and len(bank_check) > 0:
            limits['银行'] = bank_check
        if airline_check is not None and len(airline_check) > 0:
            limits['航空公司'] = airline_check
        if price_check is not None and len(price_check) > 0:
            limits['价格'] = price_check
        if time_check is None and loc_check is None and cur_check is None and bank_check is None and airline_check is None and price_check is None:
            return None

        return limits
        

if __name__ == '__main__':
    # limiter = Limiter('北京国际机场航站楼在上午七点之后还能通宵在5号航站楼到12点之前手淫吗')
    limiter = Limiter('T2航站楼十点重2号航站楼3楼7FB18层')
    print(limiter.check())
    print(limiter.sent)
