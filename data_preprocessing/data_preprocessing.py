#将知识库即ICD10中实体和ID，爱尔眼科数据存储为bioel需要的形式
import pandas as pd 
import json
import math
import sys
sys.path.append("../")
from utils import Logger
logger=Logger("../data/processed_data/aier_data/new_data_Stastics.log",on=True)

def read_kb():
    data=pd.read_excel("../data/initial_data/aier_data/disease.xlsx",sheet_name="诊断术语标准")
    return data

def read_data():
    data=pd.read_excel("../data/initial_data/aier_data/disease.xlsx",sheet_name="疾病待checkV2.0（标注全）")
    return data

def process_kb(kb:pd.DataFrame) -> map:
    entityKb={}
    # entityID={}
    #entityKb中存储的是id:实体，entityID中存储的是实体：id
    for line in kb.itertuples():
        entityKb[line.疾病编码]=[line.疾病名称]
        # entityID[line.疾病名称]=[line.疾病编码]
    logger.log(f"知识库中的实体数为: {len(entityKb)}")
    # logger.log(f"entityID中的实体数为: {len(entityID)}")
    return entityKb#,entityID

def process_data(data:pd.DataFrame,entityKb:map) -> tuple[list,list,map]:
    data=data[['疾病','标准术语']][:599]
    data.dropna(axis=0,subset = ["标准术语"],inplace=True)
    data=data.drop_duplicates(subset=['疾病'],keep=False)
    logger.log(f"去掉重复的术语和无标准术语后剩余的mention数为：{len(data)}")

    diseases=[line.疾病 for line in data.itertuples()]
    standards=[line.标准术语 for line in data.itertuples()]
    # data={disease:entityID[standard] for disease,standard in zip(diseases,standards)}
    #diseases中存储的是mention，satandards中存储的是mention对应的entity，data中存储的是mention:entity对应的id
    data={}

    for i in range(len(diseases)):
        disease=diseases[i]
        standard=standards[i].strip(" \n，")

        if disease=="子宫腺肌症":
            standard="子宫腺肌病"
        for key in entityKb.keys():           
            if entityKb[key][0]==standard:
                id=key
                break
        data[disease]=id

    return diseases,standards,data

def get_contexts(path) ->list :
    with open(path,encoding="utf-8") as f:
            dataSplit=[json.loads(line) for line in f]
    contexts=[line.replace("/n","") for line in dataSplit]
    return contexts

def get_context_for_mention(diseases:list,contexts:list) -> map:
    mentionContext={}
    for disease in diseases:
        for context in contexts:
            if disease in context:
                mentionContext[disease]=context
    if len(diseases)==len(mentionContext):
        logger.log(f"所有的mention都找到了对应的context")
    else:
        logger.log("存在部分mention未找到context")
        noContextMention=[mention for mention in diseases if  not mention in mentionContext.keys()]
        logger.log(f"这些mention为{','.join(noContextMention)}")

    return mentionContext

def process_context(disease:str,context:str):
    startIndex=context.find(disease)
    endIndex=startIndex+len(disease)
    context1=context[:startIndex]
    context2=context[startIndex:endIndex]
    context3=context[endIndex:]
    newContext=context1+"[E1]"+context2+"[/E1]"+context3
    return newContext   

def construct_dataset(diseases:list,mentionID:map,mentionContext:map) -> list:

    samples=[]
    for disease in diseases:
        sample={}
        subSample={}
        if disease not in mentionContext.keys():
            context="[E1]"+disease+"[/E1]"
        else:
            context=process_context(disease,mentionContext[disease])
        id=mentionID[disease]
        subSample["mention"]=disease
        subSample["kb_id"]=id

        sample["text"]=context
        sample["mention_data"]=[subSample]
        samples.append(sample)
    logger.log(f"构建的数据集的形式如下:\n{samples[0]}")    
    return samples

def save_kb(entityKb:map,savePath):
    with open(savePath,"w",encoding="utf-8") as f:
        json.dump(entityKb,f,ensure_ascii=False)

def save_dataset(samples:list,savePath):
    jsonLines="\n".join(json.dumps(item,ensure_ascii=False) for item in samples)
    with open(savePath,'w',encoding="utf-8") as f:
        f.write(jsonLines)

def main():
    
    kb=read_kb()
    entityKb=process_kb(kb)
    kbSavePath="../data/processed_data/aier_data/entity_kb.json"
    save_kb(entityKb,kbSavePath)

    data=read_data()
    contextPath="../data/initial_data/aier_data/入院记录数据/contextdata.json"
    contexts=get_contexts(contextPath)
    diseases,standards,mentionID=process_data(data,entityKb)
    mentionContext=get_context_for_mention(diseases,contexts)
    dataset=construct_dataset(diseases,mentionID,mentionContext)

    trainDataset=dataset[:math.floor(len(dataset)*0.6)]
    devDataset=dataset[math.floor(len(dataset)*0.6):math.floor(len(dataset)*0.8)]
    testDataset=dataset[math.floor(len(dataset)*0.8):]

    trainPath="../data/processed_data/aier_data/train.json"
    devPath="../data/processed_data/aier_data/dev.json"
    testPath="../data/processed_data/aier_data/test.json"
    save_dataset(trainDataset,trainPath)
    save_dataset(devDataset,devPath)
    save_dataset(testDataset,testPath)

    logger.log(f"训练集的样本数为:{len(trainDataset)}")
    logger.log(f"验证集的样本数为:{len(devDataset)}")
    logger.log(f"测试集的样本数为:{len(testDataset)}")

if __name__=='__main__':
    main()




    


        







