#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multi-turn synthetic customer↔merchant email threads to JSONL.

Usage examples:

# OpenAI (needs OPENAI_API_KEY)
python generate_email_corpus.py \
  --provider openai \
  --model gpt-4o-mini \
  --count 100 --min-turns 3 --max-turns 5 \
  --outdir Jupiter/data/raw

# Hugging Face local (auto-downloads from HF Hub)
python generate_email_corpus.py \
  --provider hf \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --count 50 --min-turns 3 --max-turns 5 \
  --outdir Jupiter/data/raw
"""

from __future__ import annotations
import os, re, sys, time, json, argparse, random, datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # dotenv not installed, will rely on system environment variables

# ---------- small utilities ----------
def post_clean(s: str) -> str:
    """Light cleanup for model outputs."""
    s = s.strip()
    # Remove duplicated closing prompts or extra quotes
    s = s.replace("\r\n", "\n").strip(" \n\"'")
    return s

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

# ---------- LLM adapters ----------
class LLM:
    def generate(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
        raise NotImplementedError

class OpenAILLM(LLM):
    def __init__(self, model: str, temperature: float = 0.7, max_new_tokens: int = 220):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing openai package. Install with: pip install openai") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI()
        self.model = model
        self.default_temp = temperature
        self.default_max = max_new_tokens

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        t  = self.default_temp if temperature    is None else temperature
        mx = self.default_max  if max_new_tokens is None else max_new_tokens
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=t,
            max_tokens=mx,
        )
        text = resp.choices[0].message.content or ""
        return post_clean(text)

class HFLocalLLM(LLM):
    def __init__(self, model: str, device: Optional[str] = None, temperature: float = 0.7, max_new_tokens: int = 220):
        try:
            from transformers import pipeline  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "transformers not found. Install with:\n"
                "pip install transformers accelerate sentencepiece bitsandbytes"
            ) from e
        from transformers import pipeline
        # device_map="auto" chooses GPU if available; fallbacks to CPU
        self.pipe = pipeline("text-generation", model=model, device_map="auto" if device is None else device)
        self.default_temp = temperature
        self.default_max  = max_new_tokens

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        t  = self.default_temp if temperature    is None else temperature
        mx = self.default_max  if max_new_tokens is None else max_new_tokens
        out = self.pipe(prompt, do_sample=True, temperature=t, max_new_tokens=mx)
        text = out[0]["generated_text"]
        # Some HF models echo the prompt; strip it if present
        body = text[len(prompt):] if text.startswith(prompt) else text
        return post_clean(body)

# ---------- domain data ----------
PRODUCTS = ["Bluetooth headphones", "USB-C hub", "Yoga mat", "Desk lamp", "4K monitor", "Sneakers"]
COURIERS = ["UPS", "USPS", "FedEx", "DHL"]

PROMPT_LIBRARY = {
    "topics": [
        {"name": "shipping_delay", "weight": 2.0, "labels": ["shipping", "delay"]},
        {"name": "refund_request", "weight": 1.6, "labels": ["refund"]},
        {"name": "exchange_size",  "weight": 1.2, "labels": ["exchange", "apparel"]},
        {"name": "wrong_item",     "weight": 1.0, "labels": ["wrong_item"]},
        {"name": "invoice_request","weight": 0.8, "labels": ["invoice", "billing"]},
        {"name": "bulk_quote",     "weight": 0.6, "labels": ["b2b", "quote"]},
    ],
    "customer_openers": {
        "shipping_delay": [
            "You are a customer. Write a concise, polite email asking why the order placed {days_ago} days ago hasn't updated in tracking. Include order ID #{order_id} and courier {courier}.",
            "Act as a frustrated but respectful buyer. Your express shipment via {courier} shows 'in transit' for {days_ago} days with no movement. Ask for ETA. Order #{order_id}.",
        ],
        "refund_request": [
            "You bought a product that doesn't meet expectations. Ask for a refund and outline the issue briefly. Include order #{order_id} for {product}.",
        ],
        "exchange_size": [
            "You received shoes that don't fit. Ask for an exchange to size {size}. Provide order #{order_id} politely.",
        ],
        "wrong_item": [
            "You received the wrong color/model. Request replacement or refund. Mention order #{order_id} and the intended {product}.",
        ],
        "invoice_request": [
            "You need a formal invoice for reimbursement. Ask the merchant to send a PDF invoice for order #{order_id} with company details.",
        ],
        "bulk_quote": [
            "You're a procurement manager. Request a wholesale quote for {qty} units of {product}. Ask about lead time and payment terms.",
        ],
    },
    "merchant_style": [
        (
            "You are a merchant support agent. Reply to the customer's email below in ≤180 words. "
            "Follow: (1) acknowledge & brief apology if applicable; (2) specific resolution or ETA; "
            "(3) ask for any missing info; (4) friendly closing with 'Support Team'.\n\n"
            "Customer email:\n\"\"\"{customer_email}\"\"\"\n"
        ),
        (
            "As a support specialist, respond clearly and helpfully. If order present, restate order id, courier, and ETA. "
            "If refund/exchange, outline the steps (labels/RMA).\n\n"
            "Customer email:\n\"\"\"{customer_email}\"\"\"\n"
        ),
    ],
}

SUBJECT_TEMPLATES = {
    "shipping_delay": ["Where is my order?", "Tracking not updating for order #{order_id}"],
    "refund_request": ["Refund request for order #{order_id}"],
    "exchange_size":  ["Request to exchange size for order #{order_id}"],
    "wrong_item":     ["Wrong item received - order #{order_id}"],
    "invoice_request":["Invoice request for order #{order_id}"],
    "bulk_quote":     ["Wholesale inquiry: {product} ({qty} units)"],
}

# ---------- data classes ----------
@dataclass
class Turn:
    role: str           # 'customer' | 'merchant'
    subject: str
    body: str
    scsa: Optional[str] = None

@dataclass
class Thread:
    thread_id: str
    topic: str
    labels: List[str]
    turns: List[Turn]
    meta: Dict

# ---------- helpers ----------
def sample_topic() -> Tuple[str, List[str]]:
    topics = PROMPT_LIBRARY["topics"]
    weights = [t["weight"] for t in topics]
    choice = random.choices(topics, weights=weights, k=1)[0]
    return choice["name"], choice["labels"]

def render_subject(topic: str, **vars) -> str:
    cand = random.choice(SUBJECT_TEMPLATES[topic])
    return cand.format(**vars)

def render_customer_open(topic: str, **vars) -> str:
    templ = random.choice(PROMPT_LIBRARY["customer_openers"][topic])
    return templ.format(**vars)

def render_merchant_reply(customer_email: str) -> str:
    templ = random.choice(PROMPT_LIBRARY["merchant_style"])
    return templ.format(customer_email=customer_email)

def scsa_summarize(llm: LLM, text: str, max_new_tokens: int = 110) -> str:
    """Small structured abstract (model-agnostic, deterministic-ish)."""
    prompt = (
        "Summarize the following email into a compact structured abstract (≤60 words). "
        "Include: intent, key entities (order id, product), requested action, and sentiment.\n\n"
        f"EMAIL:\n\"\"\"{text}\"\"\"\n"
        "STRUCTURED ABSTRACT:"
    )
    try:
        return llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.2)
    except Exception:
        return text.strip().split("\n")[0][:200]

# ---------- main generator ----------
def generate_thread(llm: LLM, idx: int, min_turns: int = 3, max_turns: int = 5) -> Thread:
    total_turns = random.randint(max(2, min_turns), max(min_turns, max_turns))
    topic, labels = sample_topic()

    order_id = f"{random.randint(100000, 999999)}"
    product = random.choice(PRODUCTS)
    courier = random.choice(COURIERS)
    size = random.choice(["7","8","9","M","L"]) if topic == "exchange_size" else ""
    days_ago = random.choice([2,3,4,5,7])
    qty = random.choice([10,25,50,100])

    cust_prompt = render_customer_open(topic, order_id=order_id, days_ago=days_ago, size=size, qty=qty, product=product, courier=courier)
    cust_email = llm.generate(cust_prompt, max_new_tokens=220)
    subject = render_subject(topic, order_id=order_id, qty=qty, product=product)

    turns: List[Turn] = [Turn(role="customer", subject=subject, body=cust_email)]
    last_merchant_reply = ""

    while len(turns) < total_turns:
        if turns[-1].role == "customer":
            prompt = (
                "You are a merchant support agent. Reply professionally to the customer's latest email. "
                "If shipping delay: provide ETA and apologize. If exchange/refund: describe steps (labels, RMA). "
                "If invoice: confirm and promise PDF. Keep ≤180 words.\n\n"
                f"Customer email:\n\"\"\"{turns[-1].body}\"\"\"\n"
            )
            reply = llm.generate(prompt, max_new_tokens=220)
            turns.append(Turn(role="merchant", subject=f"Re: {subject}", body=reply))
            last_merchant_reply = reply
        else:
            prompt = (
                "You are the same customer. Write a brief follow-up to the merchant's last reply (≤80 words). "
                "If info was requested (order id/photos/address), provide it. If an ETA was given, ask a clarifying question or confirm understanding.\n\n"
                f"Merchant reply:\n\"\"\"{last_merchant_reply}\"\"\"\n"
            )
            follow = llm.generate(prompt, max_new_tokens=140)
            turns.append(Turn(role="customer", subject=f"Re: {subject}", body=follow))

    # Add SCSA summaries per turn
    for t in turns:
        t.scsa = scsa_summarize(llm, t.body)

    thr = Thread(
        thread_id=f"thr_{idx:06d}",
        topic=topic,
        labels=labels,
        turns=turns,
        meta={
            "order_id": order_id,
            "product": product,
            "courier": courier,
            "generation_model": getattr(llm, "model", getattr(llm, "__class__", type("x",(object,),{})).__name__),
            "created_at": now_iso(),
            "turn_count": len(turns),
        },
    )
    return thr

# ---------- main CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate multi-turn email threads to JSONL.")
    ap.add_argument("--provider", choices=["openai","hf"], required=True, help="LLM backend")
    ap.add_argument("--model", required=True, help="Model name (e.g., gpt-4o-mini or mistralai/Mistral-7B-Instruct-v0.3)")
    ap.add_argument("--count", type=int, default=50, help="Number of threads")
    ap.add_argument("--min-turns", type=int, default=4, help="Min emails per thread")
    ap.add_argument("--max-turns", type=int, default=6, help="Max emails per thread")
    ap.add_argument("--outdir", default="Jupiter/data/raw", help="Directory to save JSONL")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)

    # init llm
    if args.provider == "openai":
        llm: LLM = OpenAILLM(model=args.model, temperature=0.7, max_new_tokens=220)
    else:
        llm = HFLocalLLM(model=args.model, temperature=0.7, max_new_tokens=220)

    # prepare output
    os.makedirs(args.outdir, exist_ok=True)
    date_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(args.outdir, f"threads_{args.provider}_{date_tag}.jsonl")

    print(f"🚀 Starting generation: provider={args.provider}, model={args.model}")
    print(f"   count={args.count}, turns={args.min_turns}-{args.max_turns}")
    print(f"   output={out_path}")

    import jsonlines  # lazy import for clarity
    ok = 0
    with jsonlines.open(out_path, mode="w") as writer:
        for i in range(args.count):
            for attempt in range(4):
                try:
                    thr = generate_thread(llm, idx=i, min_turns=args.min_turns, max_turns=args.max_turns)
                    writer.write(asdict(thr))
                    ok += 1
                    if (i+1) % 10 == 0 or (i+1) == args.count:
                        print(f"✅ generated {i+1}/{args.count}")
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[warn] sample {i} attempt {attempt+1} failed: {e} → retry in {wait}s")
                    time.sleep(wait)
                    if attempt == 3:
                        print(f"[error] sample {i} failed after retries: {e}")

    print(f"🎉 Done. Wrote {ok} threads → {out_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
