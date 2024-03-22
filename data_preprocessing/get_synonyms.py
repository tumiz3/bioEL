#从bios知识图谱的包含中英文术语的实体中获得其中所有的中文术语，保存其术语名及CID，其中CID相同的术语互为同义词
import pickle
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import Logger

def read_concepts(concept_path):
    print("开始读文件")
    with open(concept_path,encoding="utf-8") as f:
        data=f.readlines()
    print("读完文件")
    return data

def process_data(data):
#从所有的实体中获得有同义词的实体，synonyms的键为CID，值为同义词的列表
    synonyms={}
    for line in tqdm(data):
        if not ENG_or_CHS(line):
            continue
        else:
            CID,STR=get_english_concepts(line)
        if CID in synonyms.keys():
            synonyms[CID].append(STR)
        else:
            synonyms[CID]=[]
            synonyms[CID].append(STR)

    return synonyms

def ENG_or_CHS(concept):
    #用于判断一个术语是中文还是英文，如果是英文则跳过，是中文则保存
    if "ENG" in concept:
        return False
    return True

def get_english_concepts(concept):
    #获得一个术语的CID和STR，CID相同为同义词，STR为术语名
    content=concept.split("|")
    CID=content[0]
    STR=content[2]
    
    return CID,STR

def filter_synonyms(synonyms):
    #从同义词列表中取出仅包含一个同义词实体
    noSynonymKey=[]
    lenOfSynonyms=0
    print("开始处理同义词，去除仅有一个同义词的实体")
    for key in tqdm(synonyms.keys()):
        if len(synonyms[key])<2:
            noSynonymKey.append(key)
    #获得仅有一个同义词的实体的CID
        else:
            lenOfSynonyms+=len(synonyms[key])
            
    for key in noSynonymKey:
        synonyms.pop(key,None)

    return synonyms,lenOfSynonyms

def save_synonyms(synonyms):
    with open("../data/processed_data/external_data/synonyms.pickle","wb") as f:
        pickle.dump(synonyms,f)

def main():
    logger=Logger("../data/processed_data/external_data/synonyms.log",on=True)
    print("处理bios知识图谱中的所有实体，从中获得具有中文同义词的中文实体/术语")
    data=read_concepts("../data/initial_data/external_data/ConceptTerms.txt")
    logger.log(f"处理前bios中的实体名共有{len(data)}个")
    synonyms,lenOfSynonyms=filter_synonyms(process_data(data))
    logger.log(f"处理后共有{len(synonyms)}个实体,{lenOfSynonyms}个同义词")
    save_synonyms(synonyms)

if __name__ == '__main__':
    main()