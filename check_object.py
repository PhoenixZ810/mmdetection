import time


import spacy
import cupy
import en_core_web_trf

all_labels_trf = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]
'''
    PERSON：人名
    NORP：国籍或政治团体
    FAC：建筑物、机场、高速公路、桥梁等
    ORG：组织、公司、机构、部门等
    GPE：国家、城市、州等
    LOC：非政治性的地理区域，如山脉、水域等
    PRODUCT：产品名称，如汽车、食品、服装等，产品型号名
    EVENT：事件名称，如战争、运动赛事、节日等
    WORK_OF_ART：艺术作品，如书籍、歌曲、电影等
    LAW：法律文件，如法案、条约、法规等
    LANGUAGE：语言名称，如英语、汉语、拉丁语等
    DATE：日期，如2019年、10月1日、明天等
    TIME：时间，如上午10点、晚上8点、两小时等
    PERCENT：百分比，如50%、百分之一等
    MONEY：货币值，如100美元、一亿欧元等
    QUANTITY：数量，如20公里、15吨、一打等
    ORDINAL：序数，如第一、第二、第十等
    CARDINAL：基数，如一、二、十、一百等
'''
passive_entity_labels = [
    "EVENT",
    "FAC",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "WORK_OF_ART",
]
ban_entity_labels = ["CARDINAL", "DATE", "GPE", "LANGUAGE", "ORDINAL", "ORG", "PERCENT", "TIME"]
doubt_entity_labels = ["LAW", "NORP", "WORK_OF_ART"]
# spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")
# nlp = en_core_web_trf.load()
start_time = time.time()
sentence = "Sophomore Josh Alvarez is seen resting his head while watching a video during his history class; school has only been in session for a month and a half, and students are already tired (Joslyn Bowman)"
# 'So humble was Abraham Lincoln that he preferred to live in a tent on the White House lawn'
# doc = nlp("parents with their child in front of eiffel tower")
# doc = nlp("Jack Nicklaus tightening his glove in preparation of a drive during the 1981 US Open")
doc = nlp(sentence)

# print([(w.text, w.pos_) for w in doc])
# name = "Jack Nicklaus"
ents = [token for token in doc.ents]
for n in ents:
    print("ents:", n.text, n.label_)
sents = [token for token in doc.sents]
print("sents:", sents)
# print(names)
# for token in doc:
#     if token.ent_type_ == "PERSON":
#         print(token.text, "是人名")
#     else:
#         print(token.text, "不是人名")
# print(doc.ents)
# if name in names:
#     print(name)
# else:
#     print("not found")
# # print([w for w in doc.ents])
# end_time = time.time()
# print("程序运行时间：", end_time - start_time, "秒")
