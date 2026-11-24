"""Pack Inkling image requests for the native vision encoder."""

import ctypes


def image_messages(messages, images):
    kinds = ("image", "image_url", "input_image")
    markers = sum(part.get("type") in kinds for msg in messages if isinstance(msg.get("content"), list)
                  for part in msg["content"] if isinstance(part, dict))
    if markers == 0 and images and messages and messages[-1].get("role") == "user":
        content = messages[-1]["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        messages = [*messages[:-1], {**messages[-1], "content": [{"type": "image"} for _ in images] + content}]
    elif markers != len(images):
        raise ValueError("Inkling image count does not match the conversation")
    return messages


def image_payload(images, offsets):
    modes = ["RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
             for image in images]
    total = ctypes.sizeof(offsets) + sum(image.width * image.height * len(mode)
                                         for image, mode in zip(images, modes))
    if total > 0x7fffffff:
        raise ValueError("Inkling image payload exceeds the native tensor size limit")

    data = bytearray(bytes(offsets))
    tensors = [dict(name="image_offsets", dtype="int32", shape=[len(offsets)], offset_bytes=0, nbytes=len(data))]
    for image, mode in zip(images, modes):
        if image.mode != mode:
            image = image.convert(mode)
        pixels = image.tobytes()
        tensors.append(dict(name="images", dtype="uint8", shape=[image.height, image.width, len(mode)],
                            offset_bytes=len(data), nbytes=len(pixels)))
        data.extend(pixels)
    return {"mode": "inkling2", "tensors": tensors}, (ctypes.c_char * len(data)).from_buffer(data)
