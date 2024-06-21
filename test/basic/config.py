import math

default_messages_list = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": '''北京有什么景点？'''}
    ]
]

def compute_cosine_similarity(a, b):
    l = min(len(a), len(b))
    a = a[:l]
    b = b[:l]
    dot_product = sum(v1 * v2 for v1, v2 in zip(a, b))
    norm_vec1 = math.sqrt(sum(v ** 2 for v in a))
    norm_vec2 = math.sqrt(sum(v ** 2 for v in b))
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity

model_list = [
    "Qwen/Qwen2-0.5B-Instruct/",
    "Qwen/Qwen2-1.5B-Instruct/",
    "Qwen/Qwen2-7B-Instruct/"
]