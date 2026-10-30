"""Optional Nomas-PE HTTP client; inference remains in fastllm C++/CUDA.

Read the released prompt assets without importing the official Python runtime.
Only scene descriptions are enhanced; camera continuations reuse native state.
"""
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


PERSPECTIVES = ("first-person", "second-person", "third-person follow", "third-person observer")


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+:[0-5]\d\.\d{3}", value):
        raise ValueError("invalid action timestamp")
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + float(seconds)


def shape_scene(item, duration):
    """Keep the released inference fields and the current segment's actions.

    The native demo generates a fixed frame window. Future actions in a PE
    response must not be squeezed into that window or replayed on navigation.
    """
    if not isinstance(item, dict) or item.get("perspective_view") not in PERSPECTIVES:
        raise ValueError("missing or invalid perspective_view")
    if item.get("Style") not in ("real-world footage", "animation/anime", "CG", "video game",
                                 "software/web/app/computer screen recording", "slideshow"):
        raise ValueError("missing or invalid Style")
    subjects, segments = item.get("subject_list"), item.get("segments")
    if not isinstance(subjects, list) or len(subjects) != 1 or not isinstance(subjects[0], dict):
        raise ValueError("expected exactly one controlled subject")
    subject = subjects[0]
    if subject.get("subject_id") != "<|Subject1|>":
        raise ValueError("expected Subject1")
    for key in ("display_name", "subject_attribute"):
        if not isinstance(subject.get(key), str) or not subject[key].strip():
            raise ValueError("missing subject " + key)
    if not isinstance(segments, list) or len(segments) != 1 or not isinstance(segments[0], dict):
        raise ValueError("expected one scene segment")
    segment = segments[0]
    description = segment.get("visual_description_long")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("missing scene description")
    actions = segment.get("contained_actions", [])
    if not isinstance(actions, list):
        raise ValueError("contained_actions must be a list")
    start = timestamp(segment.get("start_time", "00:00.000"))
    kept = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("action_description"), str):
            raise ValueError("invalid action description")
        begin, end = timestamp(action.get("start_time")) - start, timestamp(action.get("end_time")) - start
        if begin < 0 or end <= begin:
            raise ValueError("invalid action interval")
        if begin < duration:
            kept.append({"start_time": f"00:{begin:06.3f}",
                         "end_time": f"00:{min(end, duration):06.3f}",
                         "action_description": action["action_description"]})
    # Match the released keep_keywords: segment timestamps and first-turn
    # description_change are omitted from the text encoder's input.
    shaped = {"perspective_view": item["perspective_view"], "Style": item["Style"],
              "subject_list": [{key: subject[key] for key in ("subject_id", "display_name", "subject_attribute")}],
              "segments": [{"visual_description_long": description, "contained_actions": kept}]}
    prompt = json.dumps(shaped, ensure_ascii=False)
    if len(prompt.encode()) > 16384:
        raise ValueError("expanded scene is too long")
    return prompt


class PromptEnhancer:
    def __init__(self, url, model, prompts, timeout=180):
        if not url.startswith(("http://", "https://")):
            raise ValueError("prompt enhancer URL must use HTTP or HTTPS")
        self.url, self.model, self.timeout = url, model, timeout
        directory = Path(prompts)
        self.system = (directory / "online_user_video_rewriter_system_prompt.md").read_text(encoding="utf-8")
        self.first = (directory / "online_user_video_rewriter_first_turn.md").read_text(encoding="utf-8")

    def complete(self, messages, seed, cancelled):
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "temperature": 0.2, "max_tokens": 1536, "seed": seed % (2**31),
                   "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(self.url, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        deadline = time.monotonic() + self.timeout
        parts, finish = [], None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    if cancelled():
                        raise RuntimeError("generation cancelled")
                    if time.monotonic() > deadline:
                        raise RuntimeError("prompt enhancement timed out")
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError("prompt enhancer returned an error")
                    for choice in event.get("choices", []):
                        parts.append((choice.get("delta") or {}).get("content") or "")
                        finish = choice.get("finish_reason") or finish
                    if sum(map(len, parts)) > 32768:
                        raise ValueError("prompt enhancer response is too large")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("prompt enhancement service is unavailable or timed out") from exc
        if finish != "stop":
            raise ValueError("prompt enhancer did not finish a complete response")
        content = "".join(parts).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        return json.loads(content)

    def enhance(self, caption, *, seed, num_frames, perspective="auto", previous=None, cancelled=lambda: False):
        if perspective == "auto":
            if previous:
                perspective = json.loads(previous)["perspective_view"]
            else:
                perspective = "first-person" if re.search(r"first[ -]person|第一人称", caption, re.I) else "third-person observer"
        duration = num_frames / 24
        rules = (f"Generate one immediate scene window lasting {duration:.3f} seconds. "
                 "Describe only the requested subjects, equipment and environment. Preserve counts and identity. "
                 "Do not invent a sequence of future actions or camera movements. Keep the scene concise, "
                 "under 250 English words total. When translating, preserve each object's exact category "
                 "and subtype, the number of separate parts, and how equipment is worn or used. "
                 "Resolve specialized terms by their precise meaning in context; do not substitute a "
                 "related object or activity, change quantities, or duplicate the main subject. "
                 "Return raw JSON in the required schema.")
        subject, environment = caption, caption
        if previous:
            rules += (" This is a scene edit. Preserve the previous scene's static setting and identities unless "
                      "the new request changes them. Apply the new request once; do not replay previous actions. "
                      "Return the resulting scene description, not instructions to the user. "
                      "Previous scene JSON:\n" + previous)
            subject = "Apply this scene edit: " + caption
            environment = "Preserve the previous environment except for changes requested in the scene edit."
        user_input = {"perspective_view": perspective, "subject_description": subject,
                      "environment_description": environment}
        base = self.first + "\n\n" + rules + "\n\nCurrent user input JSON:\n" + json.dumps(user_input, ensure_ascii=False)
        error = None
        for attempt in range(2):
            if cancelled():
                raise RuntimeError("generation cancelled")
            text = base + ("\nReturn corrected raw JSON. Fix: " + str(error) if error else "")
            try:
                item = self.complete([{"role": "system", "content": self.system},
                                      {"role": "user", "content": text}], seed, cancelled)
                prompt = shape_scene(item, duration)
                if item["perspective_view"] != perspective:
                    raise ValueError("preserve the requested perspective_view")
                return {"model": self.model, "prompt": prompt, "source_prompt": caption, "attempts": attempt + 1}
            except (ValueError, KeyError, TypeError) as exc:
                error = exc
        raise RuntimeError("prompt enhancement returned invalid scene JSON: " + str(error))
