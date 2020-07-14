# 昆明机场知识图谱问答系统

## 使用方法

配置所需模型和资源：

- KM_KBQA/models/
    - bert_config.json
    - bert-base-chinese.bin
    - BertERCls.bin
    - check_kbqa_model.pt
- KB_KBQA/res/
    - airline.txt
    - banks.txt
    - currency.txt
    - ent_alias.txt

### 启动服务

```python
python -m KM_KBQA.qa.QAServer -lp port
```

`port`为服务器监听端口。

请求地址`http://server_address:port/?q=question`，返回一个json数组，最多有3个元素。

例如，请求`http://192.168.3.195:9899/?q=哪有吃饭的地方？`，返回结果`["您好，桥香园(F4层)在昆明长水机场出发大厅F4层", "您好，兰州拉面在15号登机口旁", "您好，兰州拉面在A值机岛岛头左手边"]`。

### 本地测试
```python
python test.py
```

自动测试`test.py`中指定的文件包含的问题，存储对应的答案。

## 代码结构

- KM_KBQA
    - common    公共基础工具，包括服务器请求，BERT，分词，拒识模型
    - config    设置文件
    - linking   实体识别组件
    - qa    问答相关模块，包括关系提取，限制提取，问答流程等
    - BertEntityRelationClassification  基于BERT的实体识别和关系分类模型
    - models    系统所需模型文件
    - res   系统所需资源文件


## 问答流程

主要问答流程实现在`KM_KBQA.qa.QA.QA`类中，核心逻辑在`QA.answer`函数中：

1. 使用替换规则修改问句并分词
2. 规则提取限制
3. 三种linker提取实体，合并排序
4. 判断是否列举型
5. 非列举型匹配关系
6. 如果没有匹配到的关系，且实体识别分值较高，算作列举
7. 匹配机场本身相关问题
8. 匹配限制
9. 答案排序
10. 生成自然语言答案 