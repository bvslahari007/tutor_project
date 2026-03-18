import os
import time
import warnings
import google.generativeai as genai
from scaledown import ScaleDownCompressor
from dotenv import load_dotenv

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

sd = ScaleDownCompressor(
    target_model="gpt-4o",
    rate="auto",
    api_key=os.getenv("SCALEDOWN_API_KEY")
)


def words(txt):
    return len(txt.split()) if txt else 0


def get_answer(q, data):
    if not data:
        return {
            "answer": "No relevant content found.",
            "before": 0,
            "after": 0,
            "saved": 0,
            "raw": "",
            "compressed": "",
            "ok": False
        }

    raw = "\n\n".join(d.get("text", "") for d in data)
    w1 = words(raw)

    try:
        result = sd.compress(context=raw, prompt=q)
        comp = result.compressed_context
        if not comp or words(comp) < 15:
            comp = "\n\n".join(d.get("text", "") for d in data[:2])
    except Exception as e:
        print(f"ScaleDown error: {e}")
        comp = "\n\n".join(d.get("text", "") for d in data[:2])

    w2 = words(comp)
    saved = round((1 - w2 / max(w1, 1)) * 100, 1)

    p = (
        f"You are a 10th grade science tutor. "
        f"Use only the text below to answer the question. "
        f"Keep the answer to 3 to 5 sentences. "
        f"If the answer is not in the text say you do not have that information.\n\n"
        f"Text:\n{comp}\n\n"
        f"Question: {q}\n\nAnswer:"
    )

    max_retries = 3
    ans = ""
    error_msg = None

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                p,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=250,
                    temperature=0.3
                )
            )
            ans = response.text
            error_msg = None
            break
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(15 * (attempt + 1))
                    continue
            break

    if error_msg:
        return {
            "answer": "Error: " + error_msg,
            "before": w1,
            "after": w2,
            "saved": saved,
            "raw": raw,
            "compressed": comp,
            "ok": False
        }

    return {
        "answer": ans,
        "before": w1,
        "after": w2,
        "saved": saved,
        "raw": raw,
        "compressed": comp,
        "ok": True
    }


if __name__ == "__main__":
    test_data = [
        {
            "text": "Photosynthesis is the process by which green plants use sunlight to make food. It takes place in the chloroplasts. Plants take in carbon dioxide and water for this process. Chlorophyll absorbs the light energy needed.",
            "score": 0.9
        },
        {
            "text": "During photosynthesis oxygen is released as a byproduct. The glucose produced is used by the plant for energy and growth.",
            "score": 0.75
        },
        {
            "text": "The French Revolution started in 1789 and changed the political structure of France.",
            "score": 0.05
        },
        {
            "text": "The Calvin cycle fixes carbon dioxide into organic molecules using energy from ATP.",
            "score": 0.65
        }
    ]

    q = "What is photosynthesis and where does it happen?"
    out = get_answer(q, test_data)

    if not out["ok"]:
        print("Error:", out["answer"])
    else:
        print("Answer:", out["answer"])
        print("Words before:", out["before"])
        print("Words after:", out["after"])
        print("Reduction:", out["saved"], "%")
        print("Compressed text:", out["compressed"])