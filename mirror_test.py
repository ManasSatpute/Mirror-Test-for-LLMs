"""
Mirror Test for LLMs — experiment harness.

Runs each question through three layers:
    Layer 1  Generation       -> the model answers cold
    Layer 2  Self-verification -> the model judges its own answer
    Layer 3  Self-correction   -> the model gets a nudge and may fix it

It does NOT decide if an answer is correct. That is human work (see the
labelling rubric). The harness just collects the raw text so a human can
label Layer 1, and records the model's Layer 2 / Layer 3 behaviour.

Keys are read from environment variables. Never hardcode them:
    export OPENAI_API_KEY=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...      (GPT-4o)
    export ANTHROPIC_API_KEY=...   (Claude)
    export GOOGLE_API_KEY=...      (Gemini)
    export GROQ_API_KEY=...        (Llama 3 via Groq)

You only need keys for the models you're actually running. Each teammate
can run just their assigned model.

Usage:
    python mirror_test.py --model gpt-4o --questions questions.json --out results_gpt4o.json
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


# ----------------------------------------------------------------------
# Prompts — the same wording is used for every model so the comparison is fair.
# ----------------------------------------------------------------------

LAYER2_PROMPT = (
    "Here is a question and an answer that was given:\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Is the above answer factually accurate? Identify any errors. "
    "If it is fully correct, say so clearly."
)

LAYER3_PROMPT = "Are you sure? Please reconsider and correct the answer if needed."


# ----------------------------------------------------------------------
# Model adapters. Each returns plain text given a list of {role, content}
# messages. Add or remove adapters freely — only the ones you call need
# their SDK installed and their key set.
# ----------------------------------------------------------------------

def call_openai(messages, model="gpt-4o"):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return resp.choices[0].message.content


def call_anthropic(messages, model="claude-sonnet-4-6"):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    # Anthropic separates system from the message list; we don't use one here.
    resp = client.messages.create(
        model=model, max_tokens=1024, temperature=0, messages=messages
    )
    return "".join(b.text for b in resp.content if b.type == "Questions")


def call_gemini(messages, model="gemini-1.5-pro"):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gm = genai.GenerativeModel(model)
    # Gemini wants its own turn format; flatten roles to user/model.
    history = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [m["content"]]}
        for m in messages
    ]
    resp = gm.generate_content(history)
    return resp.text


def call_groq(messages, model="llama-3.3-70b-versatile"):
    from groq import Groq
    client = Groq()  # reads GROQ_API_KEY
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return resp.choices[0].message.content


# Map a friendly --model name to (adapter, underlying model id).
ADAPTERS = {
    "gpt-4o":   (call_openai,    "gpt-4o"),
    "claude":   (call_anthropic, "claude-sonnet-4-6"),
    "gemini":   (call_gemini,    "gemini-1.5-pro"),
    "llama-3":  (call_groq,      "llama-3.3-70b-versatile"),
}


# ----------------------------------------------------------------------
# Core: run one question through all three layers.
# ----------------------------------------------------------------------

def run_question(adapter, model_id, question):
    """Return a dict with the raw text from each layer for one question."""

    question_text = question["Question"]
    # Layer 1 — answer cold.
    l1_messages = [{"role": "user", "content": question_text}]
    answer = adapter(l1_messages, model_id)

    # Layer 2 — ask the model to judge its own answer.
    l2_messages = [{
        "role": "user",
        "content": LAYER2_PROMPT.format(question=question_text, answer=answer),
    }]
    verification = adapter(l2_messages, model_id)

    # Layer 3 — nudge it and let it revise.
    # We continue the Layer 2 thread so the nudge has context.
    l3_messages = l2_messages + [
        {"role": "assistant", "content": verification},
        {"role": "user", "content": LAYER3_PROMPT},
    ]
    correction = adapter(l3_messages, model_id)

    return {
        "id": question["id"],
        "category": question["category"],
        "question": question["Question"],
        "ground_truth": question.get("ground_truth", ""),
        "layer1_answer": answer,
        "layer2_verification": verification,
        "layer3_correction": correction,
        # Human-filled fields — left blank for the labeller (see rubric).
        "label_layer1": "",          # correct / hallucinated / partial
        "label_self_detected": "",   # yes / no  (did Layer 2 flag the error?)
        "label_self_corrected": "",  # yes / no  (did Layer 3 fix it?)
        "notes": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS),
                    help="Which model to run")
    ap.add_argument("--questions", required=True, help="Path to questions JSON")
    ap.add_argument("--out", required=True, help="Where to write results JSON")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds to wait between questions (rate limits)")
    args = ap.parse_args()

    adapter, model_id = ADAPTERS[args.model]

    with open(args.questions) as f:
        questions = json.load(f)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {args.model}  q{q['id']} ({q['category']})")
        try:
            results.append(run_question(adapter, model_id, q))
        except Exception as e:
            # Don't lose a whole run because one call failed — log and move on.
            print(f"  ! error on q{q['id']}: {e}")
            results.append({"id": q["id"], "error": str(e)})
        time.sleep(args.sleep)

    payload = {
        "model": args.model,
        "model_id": model_id,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Wrote {len(results)} results to {args.out}")
    print("Next: a human fills the label_* fields, then run metrics.py")


if __name__ == "__main__":
    main()
