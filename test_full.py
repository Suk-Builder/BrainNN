"""BrainNN v9 完整测试脚本"""
import urllib.request, json

def call(url, data=None):
    if data:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
            headers={"Content-Type":"application/json"}, method="POST")
    else:
        req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

# 清空记忆
call("http://127.0.0.1:7862/api/reset", {})

tests = [
    ("因果推理", "因果推理是智能体理解事件先后关系的核心能力"),
    ("元学习", "元学习让智能体学习如何学习"),
    ("空间智能", "空间智能是指智能体对三维空间进行感知和推理的能力"),
    ("拓扑量子计算", "拓扑量子计算利用拓扑量子态来保护量子信息"),
    ("神经形态计算", "神经形态计算模仿大脑神经网络实现高效信息处理"),
    ("认知架构", "认知架构是AI系统中模拟人类思维过程的软件框架"),
    ("主动推理", "主动推理认为智能体通过最小化自由能来主动选择行动"),
    ("注意力机制", "注意力机制是深度学习中的关键技术"),
    ("卷积神经网络", "卷积神经网络通过局部连接和权重共享处理图像数据"),
    ("深度学习", "深度学习通过多层神经网络自动学习数据的抽象表示"),
]

total = 0
for name, text in tests:
    r = call("http://127.0.0.1:7862/api/learn", {"text": text})
    nc = r.get("learned",{}).get("new_concepts",[])
    ok = "✓" if name in nc else "✗"
    print(f"{ok} {name}: {nc}")
    total += len(nc)

h = call("http://127.0.0.1:7862/api/health")
print(f"\n结果: 新增={total}/10, 总={h['count']}")
print(f"概念: {h['concepts']}")