#项目规划（分为几个阶段，每个阶段帮我创建一个独立的文件夹，存入对应的代码，方便我日后查看和纠正）
我想要复现一篇论文的结构："H:\FYP-RAG\文献\2025.acl-long.580.pdf"，并且做一个前端网页出来

## 实现neo4j疾病知识图谱的导入
资料：“H:\MedRAG\disease-kb\README.md”，这是六年之前的代码库，可能有些依赖已经无法使用，你要根据disease-kb代码库的内容，实现neo4j的导入
Connection URL：neo4j://127.0.0.1:7687
username:neo4j
password:qingquan666

## 构建模型结构
我已经做好了W2NER模型的训练（路径："G:\W2NER\output"，测试代码路径："H:\FYP-RAG\RAG-Medical\NER_Module\test_trained_model.py"），你需要按照文献里的结构和方法，构建出RAG问答系统
用来回答问题的大模型就调用deepseek的api，key：DEEPSEEK_API_KEY = "sk-bfd7d198f966480e99e2106d4e464b32"


##做出可视化前端
一个前端网页，一个大模型对话框，一个可视化图谱显示框（注意我电脑性能，电脑是16G显存，要求是8G能跑，然后能立体点最好，至少要流畅），图谱要显示出模型所检索出来的相关内容（具体看论文）

## 做相关的测试实验
使用deepseek，
qwen3-max（Key:sk-f16800ff4f58416c9691bf325b30eae3, api调用方式:https://bailian.console.aliyun.com/cn-beijing?tab=model#/model-market/detail/qwen3-max），
doubao（key：46d21dc9-f8e3-4e5f-9157-0db7b713011c, 
示例代码： 
import os
from openai import OpenAI

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
api_key = os.getenv('ARK_API_KEY')

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
)

response = client.responses.create(
    model="doubao-seed-2-0-pro-260215",
    input=[
        {
            "role": "user",
            "content": [

                {
                    "type": "input_image",
                    "image_url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"
                },
                {
                    "type": "input_text",
                    "text": "你看见了什么？"
                },
            ],
        }
    ]
)

print(response)
）
模型框架已经在phase2实现过了，所以可以直接用phase2的代码，但是不要动phase2文件夹的内容

实验内容： 仿照论文的Table2,Table3做实验，不用做Ablation实验和对比众多的baselines, 就和普通的KGRAG做一下对比，在MMCU-Medical,CMB-Exam,CMB-Clin数据集（"H:\MedRAG\phase4_experiments\datasets"）上测试EM,PCR,ACJ,PPL,ROUGE-R,BLEU-1,BLEU-4指标（和table2，table3一样的安排），最后的结果帮我写两个表格在一个新建文件里。 ACJ就用deepseek模型评估好了。各种设置请参考论文

论文中Metrics介绍位置： Appendix的A 2.2
提示词请准确告知大模型题目的类型，多选还是单选还是自由文本
