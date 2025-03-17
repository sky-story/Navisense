import openai
import time
import base64

client = openai.OpenAI(api_key="")

# OpenAI 费用（2024年最新）
COST_PER_1M_INPUT_TOKENS = 2.50   # 输入 token 价格 ($ per 1M tokens)
COST_PER_1M_OUTPUT_TOKENS = 10.00  # 输出 token 价格 ($ per 1M tokens)

def encode_image(image_path):
    """将本地图片转换为 Base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_paths = ["my_images/frame9.jpg", "my_images/frame10.jpg"]
image_data_list = [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
    for img_path in image_paths
]

start_time = time.time()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": (
            "You are an AI assistant designed to assist visually impaired users in navigation. "
            "Analyze two consecutive images and detect objects that might impact the user's movement, "
            "such as obstacles, vehicles, people, stairs, or sudden environmental changes. "
            "Then determine whether the user is approaching or moving away from these objects. "
            "Provide a concise and action-oriented response that helps the user navigate safely. "
            "If the user is approaching an obstacle, suggest an appropriate action like stopping, turning, or slowing down."
        )},
        {"role": "user", "content": [
            {"type": "text", "text": (
                "Analyze the following two images and detect objects that may impact the navigation of a visually impaired person. "
                "Then determine whether the user is approaching or moving away from these objects and provide a navigation suggestion."
            )},
            image_data_list[0],  
            image_data_list[1]   
        ]}
    ],
    max_tokens=1000
)

generation_time = (time.time() - start_time) * 1000  

generated_text = response.choices[0].message.content
used_input_tokens = response.usage.prompt_tokens
used_output_tokens = response.usage.completion_tokens

input_cost = (used_input_tokens / 1_000_000) * COST_PER_1M_INPUT_TOKENS
output_cost = (used_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS
total_cost = input_cost + output_cost

print("\n🔍 **Motion Analysis Result:**")
print(generated_text)

print("\n📊 **Performance Metrics:**")
print(f"⏳ Generation Time: {generation_time:.2f} ms")
print(f"📥 Input Tokens: {used_input_tokens} → Cost: ${input_cost:.5f}")
print(f"📤 Output Tokens: {used_output_tokens} → Cost: ${output_cost:.5f}")
print(f"🖼️ Image Processing Cost (2 images): ${COST_PER_IMAGE * 2:.5f}")
print(f"💰 Total API Cost: ${total_cost:.5f}")
