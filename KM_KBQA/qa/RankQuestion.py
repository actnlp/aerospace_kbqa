# from ..config import config
import re


def check_rank_questions(sent):
    def generate_list_question(limit, mention):
        chinese_keyword = ['中国', '国内', '国家', '我们']
        # 这一名单需要存储到txt中才便捷
        foreign_keyword = ['俄罗斯', '韩国', '日本', '新加坡', '巴西', '国外']
        international_keyword = ['世界', '国际']
        for keyword in chinese_keyword:
            if keyword in limit:
                return "中国有哪些" + mention
        for keyword in foreign_keyword:
            if keyword in limit:
                return "{}有哪些{}".format(keyword, mention)
        for keyword in international_keyword:
            if keyword in limit:
                return "有哪些" + mention
        return "中国有哪些" + mention

    is_rank = False
    chinese2num = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                   '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    rank_inf = {}
    re_1 = "(.*)[最](.+)的(.*)是(.*)"
    re_2 = "(.*)[第](.)(.+)的(.*)是(.*)"
    if re.match(re_1, sent) is not None:
        is_rank = True
        match_result = re.match(re_1, sent)
        rank_inf['Supplement'] = match_result.group(2)
        rank_inf['Attribute'] = match_result.group(1)
        rank_inf['Order'] = 1
        rank_inf['Generate_Question'] = generate_list_question(match_result.group(0), match_result.group(3))
    elif re.match(re_2, sent) is not None:
        is_rank = True
        match_result = re.match(re_2, sent)
        rank_inf['Supplement'] = match_result.group(3)
        rank_inf['Attribute'] = match_result.group(1)
        num = match_result.group(2)
        if num.isdigit():
            rank_inf['Order'] = int(num)
        else:
            rank_inf['Order'] = chinese2num[num]
        rank_inf['Generate_Question'] = generate_list_question(match_result.group(0), match_result.group(4))

    return is_rank, rank_inf


def select_entity(rank_inf, link_res):
    # 从属性值中得到可以排序的数值
    def get_attribute_num(attribute_value):
        numbers = re.findall(r"\d+\.?\d*", attribute_value)
        if len(numbers) == 0:
            return None
        else:
            return numbers[0]

    attribute_str = rank_inf['Attribute']
    supplement_str = rank_inf['Supplement']
    # 若一级匹配，二级匹配均未找到答案，则使用默认属性
    default_attribute = ['航站楼面积', '成立时间', '机长']
    # 一级匹配逻辑
    airport_attribute_list = {'面积': '航站楼面积', '城市': '通航城市', '旅客': '旅客吞吐量', '乘客': '旅客吞吐量',
                              '吞吐量': '货邮吞吐量', '跑道': '跑道长度', '经度': '经度', '纬度': '纬度'}

    company_attribute_list = {'员工': '员工数', '成立': '成立时间', '航线': '开通航线'}

    plane_attribute_list = {}
    # 依次分别是机场的规则替换，航空公司的规则替换，机型的规则替换
    attribute_list = [airport_attribute_list, company_attribute_list, plane_attribute_list]
    # 二级匹配逻辑
    airport_attribute_list2 = {'大': '航站楼面积', '小': '航站楼面积', '忙': '旅客吞吐量', '闲': '旅客吞吐量',
                               '北': '纬度', '南': '纬度'}

    company_attribute_list2 = {}

    plane_attribute_list2 = {'大': '机长', '长': '机长', '宽': '翼展', '重': '空机重量'}
    attribute_list2 = [airport_attribute_list2, company_attribute_list2, plane_attribute_list2]

    asc_list = ['多', '大', '长', '高', '重', '晚', '久', '忙', '紧', '北']
    desc_list = ['少', '小', '短', '低', '轻', '早', '闲', '松', '南']
    # 确定询问实体的类别
    check_type = link_res[0]['ent']['类别']
    attribute_list_index = -1
    if check_type == '国内机场' or check_type == '国外机场':
        attribute_list_index = 0
    elif check_type == '国内航空公司' or check_type == '国外航空公司':
        attribute_list_index = 1
    elif check_type == '飞机':
        attribute_list_index = 2
    # 将问题中属性信息映射到标准形式
    correct_attribute_list = attribute_list[attribute_list_index]
    correct_attribute_list2 = attribute_list2[attribute_list_index]
    flag = False
    # 从替代列表找到可以排序的属性
    for key in correct_attribute_list.keys():
        if key in attribute_str:
            attribute_str = correct_attribute_list[key]
            flag = True
    if not flag:
        for key in correct_attribute_list2.keys():
            if key in supplement_str:
                attribute_str = correct_attribute_list2[key]
                flag = True
    if not flag:
        attribute_str = default_attribute[attribute_list_index]
    # 从实体中进行匹配
    entities_with_attr = []
    for item in link_res:
        entity_can = item['ent']
        # print(entity_can)
        if attribute_str in entity_can.keys() and get_attribute_num(entity_can[attribute_str]) is not None:
            # 实体 -> 属性值中的数值
            entities_with_attr.append((entity_can, get_attribute_num(entity_can[attribute_str])))
    # print(entities_with_attr)
    if len(entities_with_attr) != 0:
        # 使用正序或者倒序对实体进行排序
        for keyword in asc_list:
            if keyword in supplement_str:
                entities_with_attr = sorted(entities_with_attr, key=lambda x: x[1], reverse=True)

        for keyword in desc_list:
            if keyword in supplement_str:
                entities_with_attr = sorted(entities_with_attr, key=lambda x: x[1])

        # 返回属性值或
        if len(entities_with_attr) >= rank_inf['Order']:
            return attribute_str, entities_with_attr[rank_inf['Order'] - 1][0], True, rank_inf['Order']
        else:
            return attribute_str, entities_with_attr[len(entities_with_attr) - 1][0], False, len(entities_with_attr)
    else:
        return attribute_str, None, True, rank_inf['Order']


def generate_answer(rank_inf):
    def get_name(ent):
        if '中文名' in ent.keys():
            return ent['中文名']
        elif 'name' in ent.keys():
            return ent['name']

    filtered_qa_res = []
    if rank_inf['Entity'] is None:
        return filtered_qa_res
    elif not rank_inf['TrueOrder']:
        temp_dict = {'natural_ans': "只为您找到数据库中排名第{}的结果：{}, 它的{}为{}".format(
            rank_inf['Order'], get_name(rank_inf['Entity']), rank_inf['Attribute'],
            rank_inf['Entity'][rank_inf['Attribute']]),
            'final_score': 0.72}
        filtered_qa_res.append(temp_dict)
        return filtered_qa_res
    else:
        temp_dict = {'natural_ans': "数据库中排名第{}的结果：{}, 它的{}为{}".format(
            rank_inf['Order'], get_name(rank_inf['Entity']), rank_inf['Attribute'],
            rank_inf['Entity'][rank_inf['Attribute']]),
            'final_score': 0.75}
        filtered_qa_res.append(temp_dict)
        return filtered_qa_res


if __name__ == "__main__":
    # print(check_rank_questions("中国面积最大的航空公司是哪个？"))
    _, airport_rank_inf = check_rank_questions("中国第二大的机场是哪个？")
    # print(airport_rank_inf)
    airport_link_res = [{'ent': {'别名': "['敦煌国际机场', '敦煌机场']", '时区': '8', '名称': "['敦煌国际机场', '敦煌机场']",
                                 '旅客吞吐量': '90.1960万人次（2019年）', '航站楼面积': '17774 m²', '修建时间': '1982年2月',
                                 '管理运营机构': '甘肃机场集团;;;甘肃省民航机场集团有限公司敦煌莫高国际机场公司(来源：百度百科)', '跑道长度': '3400米（截至2020年2月）',
                                 '机位数量': '20个（截至2020年2月）', '省份': '甘肃', '国家和地区': '中国甘肃省酒泉市敦煌市', '类别': '国内机场',
                                 '中文名': '敦煌莫高国际机场', '飞行区等级': '4D兼顾4E;;;4D级(来源：百度百科)', '地区管理': '中国民用航空西北地区管理局',
                                 '外文名': 'Dunhuang Mogao International Airport', '航线数量': '19条（截至2020年4月）', '所在城市': '敦煌',
                                 '起降架次': '8201架次（2019年）', '机场三字码': 'DNH', '通航城市': '15个（截至2020年4月）',
                                 '货邮吞吐量': '606.8吨（2019年）', '纬度': '40.17', 'name': '敦煌国际机场', '通航时间': '1982年7月',
                                 '经度': '94.81', '类型': '民用运输机场', 'ICAO机场代码': 'ZLDH', 'neoId': '89',
                                 'entity_label': 'Instance'},
                         'mention': '国内机场', 'rel_name': '类别', 'rel_val': '国内机场', 'id': '89', 'score': 1,
                         'source': 'rule'},
                        {'ent': {'别名': "['鸡西兴凯湖机场', '兴凯湖机场', '鸡西机场']", '时区': '8', '名称': "['鸡西兴凯湖机场', '兴凯湖机场']",
                                 '旅客吞吐量': '27.4482万人次（2019年）', '航站楼面积': '2588平方米（截至2020>年4月）',
                                 '管理运营机构': '首都机场集团公司\n黑龙江省机场管理集团有限公司鸡西机场分公司', '跑道信息': '2300米（截至2020年4月）',
                                 '机位数量': '4个（截至2020年4月）', '省份': '黑龙江', '国家>和地区': '中国黑龙江省鸡西市', '类别': '国内机场',
                                 '中文名': '鸡西兴凯湖机场', '飞行区等级': '4C级', '地区管理': '中国民用航空东北地区管理局',
                                 '外文名': 'Jixi Xingkaihu Airport', '航线数量': '7条（截至2020年3月）', '所在城市': '鸡西',
                                 '起降架次': '3722架次（2019年）', '机场三字码': 'JXA', '通航城市': '9个（截至2020年3月）',
                                 '货邮吞吐量': '139.2吨（2019年）', '纬度': '45.29', '通航时间': '2009年10月16日', 'name': '鸡西兴凯湖机场',
                                 '经度': '131.19', '类型': '民用运输机场', 'ICAO机场代码': 'ZYJX', 'neoId': '92',
                                 'entity_label': 'Instance'}, 'mention': '国内机场', 'rel_name': '类别', 'rel_val': '国内机场',
                         'id': '92', 'score': 1, 'source': 'rule'},
                        {'ent': {'停运时间': '2017年9月22日', '别名': "['蓝田机场', '泸州机场', '泸州蓝田机场']", '时区': '8', '名称': "['蓝田机场']",
                                 '旅客吞吐量': '98.5万人次（2016年）', '管理运营机构': '四川省机场集团', '航站楼面积': '5640 m²',
                                 '跑道信息': '2400米，共1条（截至2017年4月）', '省份': '四川', '国家>和地区': '四川省泸州市', '类别': '国内机场',
                                 '中文名': '泸州蓝田机场', '飞行区等级': '4C级', '地区管理': '中国民用航空西南地区管理局',
                                 '外文名': 'Luzhou Lantian Airport',
                                 '所在城市': '泸州', '起降架次': '9992架次（2016年）', '机场三字码': 'LZO', '通航城市': '18个（截至2017年4月）',
                                 '货邮吞吐量': '3293吨（2016年）', '纬度': '28.85', '通航时间': '1945年', 'name': '泸州蓝田机场',
                                 '经度': '105.39',
                                 '类型': '军民合用机场', 'ICAO机场代码': 'ZULZ', 'neoId': '99', 'entity_label': 'Instance'},
                         'mention': '国内机场', 'rel_name': '类别', 'rel_val': '国内机场', 'id': '99', 'score': 1,
                         'source': 'rule'},
                        {'ent': {'别名': "['德令哈机场']", '时区': '-5', '名称': "['德令哈机场']", '旅客吞吐量': '10.7563万人次（2019年）',
                                 '航站楼面积': '4400平方米（截至2020年4月>）', '管理运营机构': '西部机场集团青海机场有限公司德令哈机场分公司',
                                 '跑道信息': '3000米（截至2020年4月）', '机位数量': '4个（截至2020年4月）', '省份': '青海',
                                 '国家和地区': '中国青海省海西蒙古族藏族自治州', '类别': '国内机场', '中文名': '海西德令哈机场', '飞行区等级': '4C级',
                                 '地区管理': '中国民用航空西北地区管理局', '外文名': 'Haixi Delingha Airport', '航线数量': '3条（截至2020年3月）',
                                 '所在城市': '德令哈', '起降架次': '1889架次（2019年）', '机场三字码': 'HXD', '通航城市': '3个（截至2020年3月）',
                                 '货邮吞吐量': '1201.5吨（2019年）', '纬度': '37.07', '通航时间': '2014年6月16日', 'name': '德令哈机场',
                                 '经度': '97.16', '类型': '民用运输机场', 'ICAO机场代码': 'ZLDL', 'neoId': '160',
                                 'entity_label': 'Instance'}, 'mention': '国内机场', 'rel_name': '>类别', 'rel_val': '国内机场',
                         'id': '160', 'score': 1, 'source': 'rule'}]
    print(select_entity(airport_rank_inf, airport_link_res))
