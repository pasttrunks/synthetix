#!/usr/bin/env python3
"""
Synthetix AI Sample Generator
Generates synthetic AI text samples across multiple domains using the local Ollama API
and saves them in the Synthetix JSONL manifest format.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from typing import List, Dict, Any

# Pool of diverse domain prompts for generation
PROMPT_POOL: List[Dict[str, str]] = [
    # Academic Essay Prompts
    {
        "domain": "essay",
        "topic": "Industrial Revolution urban impact",
        "prompt": "Write an academic essay paragraph about the economic impact of the Industrial Revolution on urban working-class families in 19th-century Britain. Length: 200-300 words. Maintain a formal academic tone."
    },
    {
        "domain": "essay",
        "topic": "Photosynthetic energy conversion",
        "prompt": "Write an academic essay paragraph discussing the role of photosynthetic pigments in plant energy conversion and cellular respiration. Length: 200-300 words. Maintain a scientific, academic tone."
    },
    {
        "domain": "essay",
        "topic": "Modernist literature symbolism",
        "prompt": "Write an academic essay paragraph analyzing thematic symbolism and narrative structure in 20th-century American modernist literature. Length: 200-300 words. Maintain a formal literary critique style."
    },
    {
        "domain": "essay",
        "topic": "French Revolution causes",
        "prompt": "Write an academic essay paragraph evaluating the constitutional causes and political consequences of the French Revolution of 1789. Length: 200-300 words. Use an academic historical analysis tone."
    },
    {
        "domain": "essay",
        "topic": "Gravitational waves physics",
        "prompt": "Write an academic essay paragraph explaining the physics of gravitational waves and laser interferometer detection techniques. Length: 200-300 words."
    },

    # Casual Blog Prompts
    {
        "domain": "blog",
        "topic": "Sourdough bread baking",
        "prompt": "Write a casual, conversational blog post paragraph sharing personal experiences with learning to bake sourdough bread at home. Include enthusiastic, informal phrasing. Length: 200-300 words."
    },
    {
        "domain": "blog",
        "topic": "Minimalist home decluttering",
        "prompt": "Write a casual blog post paragraph discussing the pros and cons of switching to a minimalist lifestyle and decluttering your apartment. Length: 200-300 words."
    },
    {
        "domain": "blog",
        "topic": "Budget travel in Southeast Asia",
        "prompt": "Write a casual blog post paragraph about traveling on a budget through Southeast Asia and finding amazing street food spots. Length: 200-300 words."
    },
    {
        "domain": "blog",
        "topic": "Morning mindfulness routine",
        "prompt": "Write a friendly, personal blog post paragraph reflecting on starting a morning yoga and mindfulness routine to reduce daily stress. Length: 200-300 words."
    },

    # Business Email Prompts
    {
        "domain": "email",
        "topic": "Cybersecurity software budget request",
        "prompt": "Write a professional business email to corporate leadership requesting a budget reallocation for upgrading enterprise cybersecurity software. Include bullet points or clear justifications. Length: 150-250 words."
    },
    {
        "domain": "email",
        "topic": "Software release timeline update",
        "prompt": "Write a project manager email updating key stakeholders on software release delays due to third-party API integration issues. Maintain a clear, professional corporate tone. Length: 150-250 words."
    },
    {
        "domain": "email",
        "topic": "SaaS strategic partnership proposal",
        "prompt": "Write a formal business email proposing a strategic partnership between a SaaS company and a global logistics provider. Length: 150-250 words."
    },
    {
        "domain": "email",
        "topic": "Q4 product roadmap planning session",
        "prompt": "Write a professional email scheduling a cross-departmental product roadmap planning session for Q4 goals. Length: 150-250 words."
    },

    # Personal Narrative Prompts
    {
        "domain": "narrative",
        "topic": "Childhood summer at lakeside cabin",
        "prompt": "Write a personal narrative story about a childhood summer spent at a lakeside cabin in northern Michigan. Use rich sensory details and nostalgic phrasing. Length: 200-300 words."
    },
    {
        "domain": "narrative",
        "topic": "Moving to a new city for first job",
        "prompt": "Write a memoir-style paragraph describing the emotional experience of moving to a bustling new city for your first full-time job. Length: 200-300 words."
    },
    {
        "domain": "narrative",
        "topic": "Restoring an old vintage motorcycle",
        "prompt": "Write a personal narrative paragraph about spending weekends restoring an old vintage motorcycle in a small garage with a close friend. Length: 200-300 words."
    },
    {
        "domain": "narrative",
        "topic": "Learning traditional family recipes",
        "prompt": "Write a memoir-style paragraph about learning traditional family cooking recipes from a grandparent in a crowded kitchen. Length: 200-300 words."
    },

    # Technical Documentation Prompts
    {
        "domain": "technical",
        "topic": "Nginx reverse proxy & SSL setup",
        "prompt": "Write technical documentation explaining how to configure an Nginx reverse proxy server for SSL termination and backend load balancing. Length: 200-300 words."
    },
    {
        "domain": "technical",
        "topic": "PostgreSQL database sharding",
        "prompt": "Write technical documentation explaining database sharding strategies and horizontal scaling architecture in distributed PostgreSQL clusters. Length: 200-300 words."
    },
    {
        "domain": "technical",
        "topic": "GitHub Actions CI/CD setup",
        "prompt": "Write technical documentation detailing CI/CD pipeline setup using GitHub Actions for automated unit testing and container deployment. Length: 200-300 words."
    },
    {
        "domain": "technical",
        "topic": "JVM Garbage collection tuning",
        "prompt": "Write technical documentation explaining garbage collection algorithms and memory management parameters in high-throughput JVM applications. Length: 200-300 words."
    },

    # News Article Prompts
    {
        "domain": "news",
        "topic": "Local solar power grid installation",
        "prompt": "Write a news article paragraph reporting on municipal council approval of a major solar power grid installation project. Use standard objective journalistic style. Length: 200-300 words."
    },
    {
        "domain": "news",
        "topic": "Automotive supply chain delays",
        "prompt": "Write a news article paragraph detailing quarterly economic results and supply chain challenges facing regional manufacturing plants. Length: 200-300 words."
    },
    {
        "domain": "news",
        "topic": "Deep ocean exploration research",
        "prompt": "Write a news article paragraph covering breakthrough research findings in deep ocean hydrothermal vent exploration. Length: 200-300 words."
    },
    {
        "domain": "news",
        "topic": "Public library tech expansion",
        "prompt": "Write a news article paragraph reporting on city investments in public library digital literacy centers and public access computing. Length: 200-300 words."
    }
]

def derive_model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    if "gpt" in name_lower:
        return "gpt4"
    elif "claude" in name_lower:
        return "claude"
    elif "llama" in name_lower:
        return "llama"
    elif "gemini" in name_lower:
        return "gemini"
    else:
        return "other"

def resolve_ollama_model(base_url: str, target_model: str) -> str:
    """Fetch tags from Ollama API to match exact model tag (e.g. llama3 -> llama3:latest)."""
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(tags_url)
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode('utf-8'))
            models = [m['name'] for m in data.get('models', [])]
            if target_model in models:
                return target_model
            for m in models:
                if m.startswith(f"{target_model}:") or m.startswith(target_model):
                    return m
    except Exception:
        pass
    return target_model

def generate_sample_from_ollama(base_url: str, model: str, prompt: str) -> str:
    """Call Ollama /api/generate endpoint to obtain AI text."""
    gen_url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        gen_url,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        response_data = json.loads(res.read().decode("utf-8"))
        return response_data.get("response", "").strip()

def main():
    parser = argparse.ArgumentParser(description="Generate AI text samples using local Ollama API")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("--count", type=int, default=20, help="Number of samples to generate (default: 20)")
    parser.add_argument("--output", type=str, default="benchmark/corpus/ai_samples.jsonl", help="Output JSONL path")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Local Ollama API URL")
    parser.add_argument("--dry-run", action="store_true", help="Display prompts without invoking the API")

    args = parser.parse_args()

    # Select prompts up to requested count
    prompts_to_run = []
    for i in range(args.count):
        prompts_to_run.append(PROMPT_POOL[i % len(PROMPT_POOL)])

    if args.dry_run:
        print(f"\n[DRY RUN MODE] Generative Plan for {len(prompts_to_run)} AI Samples:")
        print(f"Target Model: {args.model}")
        print(f"Output File:  {args.output}")
        print("-" * 65)
        for idx, item in enumerate(prompts_to_run, 1):
            print(f"Sample {idx:02d} | Domain: {item['domain']:<10} | Topic: {item['topic']}")
            print(f"  Prompt: {item['prompt']}")
            print("-" * 65)
        print("\nDry run completed. No API calls executed.")
        return

    # Check Ollama connectivity & resolve exact model tag
    actual_model = resolve_ollama_model(args.ollama_url, args.model)
    model_family = derive_model_family(args.model)

    print(f"Starting AI sample generation via Ollama at {args.ollama_url}...")
    print(f"Target Model: {actual_model} (Family: {model_family})")
    print(f"Generating {len(prompts_to_run)} samples across domains...\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    generated_samples = []
    for idx, item in enumerate(prompts_to_run, 1):
        domain = item["domain"]
        topic = item["topic"]
        prompt = item["prompt"]

        print(f"[{idx}/{len(prompts_to_run)}] Generating {domain.upper()} sample on '{topic}'...", end="", flush=True)

        try:
            text = generate_sample_from_ollama(args.ollama_url, actual_model, prompt)
            word_count = len(text.split())
            print(f" Done ({word_count} words).")

            sample_entry = {
                "text": text,
                "label": "ai",
                "source": f"ollama_{args.model}",
                "domain": domain,
                "model_family": model_family,
                "word_count": word_count,
                "language": "en"
            }
            generated_samples.append(sample_entry)
        except Exception as e:
            print(f" FAILED!\nError generating sample {idx}: {e}", file=sys.stderr)
            sys.exit(1)

    # Write samples to output
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in generated_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nSuccessfully wrote {len(generated_samples)} AI samples to {args.output}")

if __name__ == "__main__":
    main()
