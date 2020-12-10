from utils import TextPairDataset
from model import BertSentSimCheckModel
import pandas as pd


# test_path = "./data/test/72_sensitive.xls"
# df = pd.read_excel(test_path)
data_total = {}
# titles = list(df["标题"])
# news = list(df["摘要"])
# labels = [0]*(len(titles))

news = [""""新京报讯（记者李一凡）今年3月，湖南常德滴滴司机陈师傅，被一名19岁的大学生乘客杀害一案，引发关注。今日（12月30日），新京报记者从常德市中院获悉，此案将于本周五（2020年1月3日）开庭。死者陈师傅的妻子表示，对方一家从未道歉，此前庭前会议中，对方提出了嫌疑人杨某淇作案时患有抑郁症的辩护意见。另具警方出具的鉴定书显示，嫌疑人作案时有限定刑事责任能力。
常德遇害司机陈师傅的妻子田女士，已收到常德中院下达的出庭传票。此案将于本周五（2020年1月3日）开庭。受访者供图
常德滴滴司机遇害案1月3日开庭
新京报此前报道，今年3月24日凌晨，滴滴司机陈师傅，搭载19岁大学生杨某淇到常南汽车总站附近。坐在后排的杨某淇趁陈某不备，朝陈某连捅数刀致其死亡。事发监控显示，杨某淇杀人后下车离开。随后，杨某淇到公安机关自首，并供述称“因悲观厌世，精神崩溃，无故将司机杀害”。据杨某淇就读学校的工作人员称，他家有四口人，姐姐是聋哑人。
今日（12月30日）下午，新京报记者从常德市中院获悉，此案将于本周五（2020年1月3日）开庭。一名工作人员表示，日前，已向家属下达了开庭传票，“开庭时间不变，以文书传达时间为准。”
新京报记者掌握的一份由常德中院于12月27日出具的《传票》显示，杨某淇被控故意杀人一案，将于1月3日9时30分，在汉寿县法院刑事审判第一庭开庭审理。
常德市鼎城区公安局出具的《鉴定意见通知书》显示，杨某淇在本案中实施危害行为时，有限定（部分）刑事责任能力。受访者供图
警方：嫌犯有限定刑事责任能力
今日（12月30日），新京报记者从陈师傅的家属处获知，陈师傅有两个儿子，大儿子今年18岁，小儿子还不到5岁。陈师傅的妻子表示，对方的家属从未道歉，此前出过几万元丧葬费，但一直说家庭经济存在困难，无法给予更多赔偿。
陈师傅妻子介绍，此案于前不久召开了庭前会议，对方提出了杨某淇作案时患有抑郁症的辩护意见。
新京报记者注意到，此案在常德警方立案侦查阶段，曾对杨某淇的精神状况进行鉴定。遇害者家属提供一份由常德市鼎城区公安局出具的鉴定意见通知书显示，该局指聘请有关人员对“被鉴定人杨某淇是否患有精神病，实施危害行为时有无刑事责任能力”，进行了“精神病和刑事责任能力鉴定”。鉴定意见显示，根据材料和检查，被鉴定人杨某淇诊断为抑郁症，在本案中实施危害行为时，有限定（部分）刑事责任能力。
校对柳宝庆"
"""]
titles = ["常德滴滴司机遇害案本周五开庭,嫌犯有限定刑事责任能力"]
labels = [0]
data_total["news"] = news
data_total["titles"] = titles
data_total["labels"] = labels

data = TextPairDataset.from_dict_list(data_total)

# true_label = data.label_list

model = BertSentSimCheckModel.load("./model")

output, loss = model.predict(data)

print(output)

from collections import Counter
print(Counter(output))
# from sklearn.metrics import classification_report
#
# report = classification_report(true_label,
#                                output,
#                                target_names=["negative", "unknown"],
#                                digits=4)
# print(report)
if __name__=="__main__":
    print("****")