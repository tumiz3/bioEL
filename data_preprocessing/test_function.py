# def process_context(disease:str,context:str):
#     startIndex=context.find(disease)
#     endIndex=startIndex+len(disease)
#     context1=context[:startIndex]
#     context2=context[startIndex:endIndex]
#     context3=context[endIndex:]
#     newContext=context1+"[E1]"+context2+"[/E1]"+context3
#     return newContext

# context="abcde"
# sub="c"
# print(process_context(sub,context))

import json
with open("../data/processed_data/aier_data/entity_kb.json",encoding="utf-8") as f:
    kb=json.load(f)
print(len(kb))