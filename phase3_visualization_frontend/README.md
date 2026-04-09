# Phase 3: 可视化前端

医学知识图谱问答系统的 Web 前端，包含大模型对话框和 3D 知识图谱可视化。

## 功能

- **大模型对话框**：输入医学问题，展示 HyKGE 生成的回答
- **2D 知识图谱**：展示检索到的 reasoning chains，以节点和边形式呈现（2D 更流畅）
- **性能优化**：限制图谱节点数（35）和边数（50），保证流畅运行

## 依赖

- Phase 1：Neo4j 疾病知识图谱已导入
- Phase 2：HyKGE RAG 系统可正常调用

## 安装与运行

```bash
cd h:\MedRAG\phase3_visualization_frontend
G:\Anaconda\envs\rag_fyp\python.exe -m pip install -r requirements.txt
G:\Anaconda\envs\rag_fyp\python.exe app.py
```

浏览器访问：http://localhost:5000

## 结构

```
phase3_visualization_frontend/
├── app.py              # Flask 后端，集成 HyKGE
├── templates/
│   └── index.html      # 主页面
├── static/
│   ├── css/style.css   # 样式
│   └── js/app.js       # 前端逻辑 + 3D 图谱
└── requirements.txt
```
