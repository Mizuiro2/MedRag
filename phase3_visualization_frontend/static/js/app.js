/**
 * Phase 3 前端 - 对话框 + 知识图谱可视化
 * 显示逻辑：锚点实体 + 重排序后的 top10 关系链（实体-关系-实体）
 */

const MAX_GRAPH_NODES = 35;
const MAX_GRAPH_LINKS = 50;

const chatMessages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const graphContainer = document.getElementById("graphContainer");
const graphHint = document.getElementById("graphHint");
const graphInfo = document.getElementById("graphInfo");

let graph = null;

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.innerHTML = `
    <div class="role">${role === "user" ? "你" : "MedRAG"}</div>
    <div class="content">${escapeHtml(content)}</div>
  `;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoadingMessage() {
  const div = document.createElement("div");
  div.id = "loadingMsg";
  div.className = "chat-message assistant loading";
  div.innerHTML = '<div class="role">MedRAG</div><div class="content">正在检索知识图谱并生成回答，请稍候…</div>';
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoadingMessage() {
  const el = document.getElementById("loadingMsg");
  if (el) el.remove();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML.replace(/\n/g, "<br>");
}

function updateGraph(graphData, chainsCount) {
  if (!graphData || !graphData.nodes || !graphData.edges) {
    graphHint.classList.remove("hidden");
    if (graphInfo) graphInfo.textContent = "";
    if (graph) {
      graph.graphData({ nodes: [], links: [] });
    }
    return;
  }

  const nodes = graphData.nodes.slice(0, MAX_GRAPH_NODES);
  const nodeIds = new Set(nodes.map((n) => n.id));
  const edges = graphData.edges
    .filter((e) => nodeIds.has(e.from) && nodeIds.has(e.to))
    .slice(0, MAX_GRAPH_LINKS);

  const gData = {
    nodes: nodes.map((n) => ({
      id: n.id,
      name: n.label,
      ...(n.title && { title: n.title }),
    })),
    links: edges.map((e) => ({
      source: e.from,
      target: e.to,
      ...(e.label && { name: e.label }),
    })),
  };

  graphHint.classList.add("hidden");
  if (graphInfo) {
    graphInfo.textContent = `共 ${nodes.length} 个实体、${edges.length} 条关系（来自 top${chainsCount || 10} 检索链）`;
  }

  if (graph) {
    graph.graphData(gData);
    setTimeout(() => graph.zoomToFit(400, 40), 100);
    return;
  }

  try {
    graph = ForceGraph()(graphContainer)
      .graphData(gData)
      .nodeLabel((n) => n.title || n.name || n.id)
      .nodeAutoColorBy("id")
      .linkColor(() => "#7ee787")
      .linkWidth(1.5)
      .linkLabel((l) => l.name || "")
      .linkDirectionalParticles(0)
      .nodeCanvasObjectMode("replace")
      .nodeCanvasObject((node, ctx, globalScale) => {
        if (node.x == null || node.y == null) return;
        const label = node.name || node.id || "";
        const fontSize = Math.max(10, 12 / globalScale);
        ctx.font = `${fontSize}px "Microsoft YaHei", sans-serif`;
        const textWidth = ctx.measureText(label).width;
        const pad = fontSize * 0.4;
        const w = textWidth + pad * 2;
        const h = fontSize + pad;
        const r = Math.sqrt(w * w + h * h) / 2;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = node.color || "#58a6ff";
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1 / globalScale;
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, node.x, node.y);
      })
      .onEngineStop(() => {
        if (graph) graph.zoomToFit(400, 50);
      });
  } catch (e) {
    console.error("图谱初始化失败:", e);
    graphHint.textContent = "图谱加载失败，请刷新页面";
    graphHint.classList.remove("hidden");
  }
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  sendBtn.disabled = true;
  addLoadingMessage();

  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await resp.json();
    removeLoadingMessage();

    if (!resp.ok) {
      addMessage("assistant", "错误: " + (data.error || "请求失败"));
      return;
    }

    addMessage("assistant", data.answer || "暂无回答");
    updateGraph(data.graph, data.chains_count);
  } catch (e) {
    removeLoadingMessage();
    addMessage("assistant", "请求失败: " + e.message);
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});
