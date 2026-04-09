# -*- coding: utf-8 -*-
"""
Phase 3: 可视化前端 - Flask 后端
集成 Phase 2 HyKGE，提供 API 给前端
"""

import os
import sys

# 将 phase2 加入路径
PHASE2_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase2_rag_system")
sys.path.insert(0, PHASE2_PATH)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# 延迟加载 HyKGE（避免启动时加载大模型）
_hykge = None


def get_hykge():
    global _hykge
    if _hykge is None:
        from hykge import HyKGE
        _hykge = HyKGE()
    return _hykge


def chains_to_graph(chains):
    """
    将 pruned_chains 转为前端图谱数据
    chains: [{"chain": str, "head": str, "tail": str, "head_desc": str, "tail_desc": str}, ...]
    返回: {nodes: [{id, label, title}], edges: [{from, to, label}]}
    """
    nodes = {}
    edges = []

    for c in chains:
        head = c.get("head", "")
        tail = c.get("tail", "")
        chain_str = c.get("chain", "")
        head_desc = c.get("head_desc", "") or ""
        tail_desc = c.get("tail_desc", "") or ""

        if not head or not tail:
            continue

        # 解析 chain 字符串获取关系类型，如 "A --[rel]--> B" 或 "A -> r1 -> B"
        rel_label = "相关"
        if "--[" in chain_str and "]-->" in chain_str:
            try:
                start = chain_str.index("[") + 1
                end = chain_str.index("]")
                rel_label = chain_str[start:end]
            except (ValueError, IndexError):
                pass
        elif " -> " in chain_str:
            parts = chain_str.split(" -> ")
            if len(parts) >= 2:
                # 取第一个关系类型
                for p in parts[1::2]:  # 奇数位是关系
                    if p and not p.startswith("["):
                        rel_label = p
                        break

        if head not in nodes:
            nodes[head] = {"id": head, "label": head, "title": head_desc[:100] if head_desc else head}
        if tail not in nodes:
            nodes[tail] = {"id": tail, "label": tail, "title": tail_desc[:100] if tail_desc else tail}

        edges.append({"from": head, "to": tail, "label": rel_label})

    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """处理用户问题，返回答案和图谱数据"""
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400

        h = get_hykge()
        result = h.query(question)

        graph_data = chains_to_graph(result.get("pruned_chains", []))

        return jsonify({
            "answer": result.get("answer", ""),
            "ho": result.get("ho", ""),
            "entities": result.get("entities", []),
            "anchors": result.get("anchors", []),
            "graph": graph_data,
            "chains_count": len(result.get("pruned_chains", []))
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
