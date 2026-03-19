import os
import google.generativeai as genai
from scaledown import ScaleDown
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
sd = ScaleDown()

def words(txt):
    return len(txt.split()) if txt else 0

def prune_and_answer(q, data):
    if not data:
        return {
            "answer": "No relevant content found.",
            "tokens_before": 0,
            "tokens_after": 0,
            "reduction_percent": 0,
            "raw_context": "",
            "compressed_context": "",
            "success": False,
            "error": "No data"
        }

    raw = "\n\n".join(d.get("text", "") for d in data)
    w1 = words(raw)

    try:
        full_prompt = f"Context:\n{raw}\n\nQuestion: {q}"
        result = sd.optimize_with_pipeline(full_prompt, optimizers=["remove_filler", "compress"])
        comp = result.get("optimized_prompt", "")
        if not comp or words(comp) < 15:
            comp = "\n\n".join(d.get("text", "") for d in data[:2])
    except:
        comp = "\n\n".join(d.get("text", "") for d in data[:2])

    w2 = words(comp)
    saved = round((1 - w2 / w1) * 100, 1) if w1 else 0.0

    p = (
        f"You are a 10th-grade science tutor. Base your answer ONLY on the text below. "
        f"Keep it to 3-5 sentences. If the info isn't there, say "
        f"'I don't have enough information in my textbooks to answer that.'\n\n"
        f"Text:\n{comp}\n\n"
        f"Question: {q}\n\nAnswer:"
    )

    try:
        response = model.generate_content(
            p,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=250,
                temperature=0.3
            )
        )
        ans = response.text

    except Exception as e:
        return {
            "answer": "Error: " + str(e),
            "tokens_before": w1,
            "tokens_after": w2,
            "reduction_percent": saved,
            "raw_context": raw,
            "compressed_context": comp,
            "success": False,
            "error": str(e)
        }

    return {
        "answer": ans,
        "tokens_before": w1,
        "tokens_after": w2,
        "reduction_percent": saved,
        "raw_context": raw,
        "compressed_context": comp,
        "success": True,
        "error": None
    }


if __name__ == "__main__":
    test_data = [
        {"text": "Photosynthesis is the process by which green plants use sunlight to make food. It takes place in the chloroplasts. Plants take in carbon dioxide and water for this process. Chlorophyll absorbs the light energy needed.", "score": 0.9},
        {"text": "During photosynthesis oxygen is released as a byproduct. The glucose produced is used by the plant for energy and growth.", "score": 0.75},
        {"text": "The French Revolution started in 1789 and changed the political structure of France.", "score": 0.05},
        {"text": "The Calvin cycle fixes carbon dioxide into organic molecules using energy from ATP.", "score": 0.65}
    ]

    q = "What is photosynthesis and where does it happen?"
    out = prune_and_answer(q, test_data)

    if not out["success"]:
        print("Error:", out["error"])
    else:
        print("Answer:", out["answer"])
        print("Words before:", out["tokens_before"])
        print("Words after:", out["tokens_after"])
        print("Reduction:", out["reduction_percent"], "%")
        print("Compressed context:", out["compressed_context"])