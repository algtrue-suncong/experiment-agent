import json
import os
from openai import OpenAI

# 用百炼替换OpenAI
client = OpenAI(
    api_key="sk-5368191a39734dbba163f7516d8bcd7c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ===== 1. 定义工具（模拟MCP思想，先本地Mock）=====
tools = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "将地址转换为经纬度坐标",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "详细地址，如'北京市海淀区中关村'"}
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_plan",
            "description": "规划两点之间的路线",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点坐标，如'116.397,39.918'"},
                    "destination": {"type": "string", "description": "终点坐标"},
                    "mode": {"type": "string", "enum": ["driving", "taxi", "bus"], "description": "出行方式"}
                },
                "required": ["origin", "destination", "mode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_price",
            "description": "估算打车费用",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_km": {"type": "number"},
                    "duration_min": {"type": "number"},
                    "car_type": {"type": "string", "enum": ["economy", "comfort", "business"], "default": "economy"}
                },
                "required": ["distance_km", "duration_min"]
            }
        }
    }
]


# ===== 2. 工具实现（Mock阶段，先不调用真实API）=====
def geocode(address: str) -> str:
    """模拟地理编码"""
    mock_db = {
        "中关村": "116.316,39.984",
        "天安门": "116.397,39.918",
        "国贸": "116.457,39.909",
        "首都机场": "116.587,40.080"
    }
    for key, val in mock_db.items():
        if key in address:
            return json.dumps({"lat": val.split(',')[1], "lng": val.split(',')[0], "name": key})
    return json.dumps({"lat": "39.9", "lng": "116.4", "name": address})  # 默认值


def route_plan(origin: str, destination: str, mode: str) -> str:
    """模拟路径规划"""
    # 简单模拟：假设直线距离10km，20分钟
    return json.dumps({
        "distance": 10.5,
        "duration": 25,
        "route": [origin, "三环辅路", "国贸桥", destination],
        "mode": mode
    })


def estimate_price(distance_km: float, duration_min: float, car_type: str = "economy") -> str:
    """模拟计价"""
    base = 14  # 起步价
    per_km = 2.5 if car_type == "economy" else 4.0
    price = base + distance_km * per_km
    return json.dumps({
        "estimate": round(price, 1),
        "currency": "CNY",
        "range": f"{round(price * 0.8, 0)}-{round(price * 1.2, 0)}"
    })


# 工具路由表
TOOL_MAP = {
    "geocode": geocode,
    "route_plan": route_plan,
    "estimate_price": estimate_price
}


# ===== 3. Agent核心循环（ReAct雏形）=====
def agent_loop(user_query: str):
    messages = [
        {"role": "system", "content": """你是智能出行助手。用户输入地点时：
1. 先用geocode解析地址坐标
2. 用route_plan规划路线
3. 如用户问价格，调用estimate_price
4. 用自然语言整合结果，给出清晰建议"""},
        {"role": "user", "content": user_query}
    ]

    # 第一轮：LLM决定用什么工具
    resp = client.chat.completions.create(
        model="qwen-plus",  # 用百炼模型
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    msg = resp.choices[0].message

    # 如果不需要工具，直接返回
    if not msg.tool_calls:
        return msg.content

    # 执行工具链（可循环多次，直到LLM说停）
    max_rounds = 5
    for _ in range(max_rounds):
        # 添加assistant的tool_calls到历史
        messages.append(msg)

        # 执行所有请求的工具
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"🔧 执行工具: {func_name}({args})")  # 调试日志

            result = TOOL_MAP[func_name](**args)

            # 将结果加入messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # 下一轮：基于工具结果生成回复或继续调用工具
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=tools
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

    return msg.content


# ===== 4. 运行测试 =====
if __name__ == "__main__":
    # 测试场景
    queries = [
        "从中关村去天安门怎么走？",
        "从国贸到机场打车多少钱？",
        "帮我规划从家（中关村）到公司（国贸）的上班路线，要便宜"
    ]

    for q in queries:
        print(f"\n👤 用户: {q}")
        print(f"🤖 Agent: {agent_loop(q)}")
        print("-" * 50)
