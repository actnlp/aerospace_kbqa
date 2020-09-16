from AsyncNeoDriver import AsyncNeoDriver


class Reader():
    def __init__(self):
        self.driver = AsyncNeoDriver.get_driver(name='default')
        self.load_all_entities()
        self.ent2alias = None
        self.ent_all_names = None
        self.alias_prop = None

        self.load_all_entities(self)

    def load_all_entities(self, entity_labels='Instance'):
        all_entities = []
        for entity_label in entity_labels:
            all_entities += self.driver.get_all_entities(entity_label).result()
        self.ent2alias, self.ent_all_names, self.alias_prop = self.get_entity_all_alias(all_entities)

    #  获取所有实体的所有名称，包括所有别名
    def get_entity_all_alias(all_entities):
        all_alias = {}
        all_names, prop_list, alias_filter = [], [], []
        alias_prop_list = ['别名', '全称', '简称', '名称', '呼号', '外文', '英文', 'IATA', 'ICAO', '三字', '二字']

        for word in alias_prop_list:
            for e in all_entities:
                alias = []
                name = e['name']
                for prop, val in e.items():
                    if word in prop:
                        if prop in ['名称', '别名'] and val[0] == "[":  # list 类型
                            val = eval(val)
                            alias += [word for word in val]
                            continue
                        alias.append(val)

                        if ";;;" in val:
                            print(val)
                        if prop not in prop_list:
                            prop_list.append(word)
                alias = list(set(alias))
                alias_filter = alias.remove(name)
            if alias_filter:
                all_alias[name] = alias_filter
                print("{}: {}".format(name, alias_filter))
            all_names += alias

        return all_alias, all_names, prop_list


if __name__ == "__main__":
    reader = Reader()
    ent2alias = reader.ent2alias
