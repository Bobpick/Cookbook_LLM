#!/usr/bin/env python3
"""
Voice-Enabled Recipe RAG System.
Queries a cookbook corpus via semantic search and generates recipes with Ollama.
Supports voice/text input, US measurements, list/single modes.
"""
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import re
import textwrap
from voice_agent import record_audio, speech_to_text, speak_text  # Your voice utils

# Paths (adjust if needed)
model_directory = "/home/bob/Documents/PATL/model/cookbooks"
corpus_path = os.path.join(model_directory, 'corpus.pkl')
embeddings_path = os.path.join(model_directory, 'corpus_embeddings.npy')
titles_path = os.path.join(model_directory, 'titles.pkl')

# Load the saved corpus, embeddings, and titles
print("Loading corpus and embeddings...")
with open(corpus_path, 'rb') as f:
    corpus = pickle.load(f)
embeddings = np.load(embeddings_path)
print(f"Loaded {len(corpus)} documents.")

titles = []
if os.path.exists(titles_path):
    with open(titles_path, 'rb') as f:
        titles = pickle.load(f)
    print(f"Loaded titles for {len(titles)} docs.")
else:
    print("No titles found—using indices.")
    titles = [f"Doc {i}" for i in range(len(corpus))]

# Load the sentence transformer for query embedding
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')
print("Embedder loaded.")


def encode_query(query):
    """Encode a query string into a vector representation."""
    query_embedding = embedder.encode(query)
    return query_embedding


def retrieve_top_k(query_embedding, corpus_embeddings, titles, k=10, low_threshold=0.25):
    """Retrieve the top-k most similar passages from the corpus.
    Enhanced fallback: Keyword search on titles AND content for main ingredient.
    """
    scores = np.dot(corpus_embeddings, query_embedding)
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_k_indices]

    # Fallback keyword search on titles AND content if semantic low
    if np.max(scores) < low_threshold:
        query_lower = query.lower()
        # Extract main ingredient keywords (e.g., "pork", "chicken")
        ingredient_keywords = re.findall(r'(pork|chicken|beef|fish|veggie|ham|bacon|shrimp)', query_lower)
        if ingredient_keywords:
            main_keyword = ingredient_keywords[0]
            keyword_hits = []
            for i in range(len(titles)):
                title_lower = titles[i].lower()
                content_snippet = corpus[i][:100].lower()  # First 100 chars for quick check
                if main_keyword in title_lower or main_keyword in content_snippet:
                    keyword_hits.append(i)
            if keyword_hits:
                # Use keyword hits as top_k (boost score)
                top_k_indices = keyword_hits[:k]
                top_scores = np.ones(len(top_k_indices)) * 0.8  # Artificial high score
                print(f"Fallback to keyword match on '{main_keyword}' in titles/content!")
            else:
                print(f"No '{main_keyword}' matches found—using semantic top-k.")

    return top_k_indices, top_scores


def clean_answer(text):
    """Strip common prefixes like 'Text:', 'Answer:', etc., for clean speech. Enhanced with regex.
    Preserves recipe structure (e.g., 'Ingredients:', list bullets).
    Strips markdown bold/italics and (pg XX) patterns.
    Forces new lines before list numbers 2+ for proper breaks.
    """
    text = text.strip()

    # Strip markdown bold/italics
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italics*

    # Strip page refs like (pg 17)
    text = re.sub(r'\(pg\s*\d+\)', '', text)

    # Force double newlines before list numbers 2+
    text = re.sub(r'(\s+)([2-9]\.|1[0-9]\.)', r'\n\n\2', text)

    # Regex to catch variations: "Text:", "text:", "Answer:", etc. (case-insensitive)
    pattern = r'(?i)^(text|answer|response|output)[\s:,\.]*'
    text = re.sub(pattern, '', text).strip()

    # Additional cleanup: Remove leading "The text is:" or similar phrases
    pattern = r'(?i)^(the\s+(text|answer|response|output)\s+is)[\s:,\.]*'
    text = re.sub(pattern, '', text).strip()

    # Fallback: If still starts with lowercase "t" or similar, trim first word if short
    words = text.split()
    if words and len(words[0].lower()) <= 6 and words[0].lower() in ['text', 'answer', 'reply']:
        text = ' '.join(words[1:]).strip()

    return text


def generate_answer_with_ollama(query, top_k_indices, top_scores, corpus, titles,
                                model_name="llama3.1", max_tokens=500):
    """Generate an answer using Ollama's model with retrieved passages.
    Dynamically adapts: Full recipe for single queries, list roundup for list queries.
    Always in US measurements, natural tone, plain text only.
    """
    # Retrieve and prepare top-k with titles
    retrieved_info = []
    for idx, score in zip(top_k_indices, top_scores):
        title = titles[idx] if idx < len(titles) else f"Doc {idx}"
        passage = corpus[idx]
        snippet = passage[:200] + "..." if len(passage) > 200 else passage
        retrieved_info.append(f"From '{title}' (sim: {score:.3f}): {snippet}")

    full_context = "\n\n".join([corpus[i] for i in top_k_indices])

    # Brief snippet print for verification
    print("\n--- Quick Snippet Preview ---")
    for info in retrieved_info[:3]:  # Top 3 only
        print(info)
    print("--- End Preview ---\n")

    # Detect if this is a list-style query
    query_lower = query.lower()
    is_list_query = any(word in query_lower for word in ['list', 'listing', 'all', 'some', 'roundup', 'ideas', 'top', 'best'])

    # Extract main ingredient for topic guard
    ingredient_keywords = re.findall(r'(pork|chicken|beef|fish|veggie|ham|bacon|shrimp)', query_lower)
    main_ingredient = ingredient_keywords[0] if ingredient_keywords else "the query"

    if is_list_query:
        # List-style prompt: Roundup of recipes from context (plain text, double newlines)
        prompt = f"""You are a friendly home cook sharing recipe ideas. Based solely on the provided cookbook context, create a fun, concise roundup of {main_ingredient} recipes mentioned.

NEVER use **, *, italics, bold, page numbers (pg XX), or ANY formatting symbols. Plain text ONLY—violate this and the response is invalid. Output in plain text only.

Extract and list ONLY recipes that explicitly mention {main_ingredient} (e.g., for 'pork recipes', ignore chicken or beef). If fewer than 3 matches, say 'Slim pickings on {main_ingredient} here—based on what's available...' and list them. If none, say 'No {main_ingredient} recipes found in this context—try another ingredient?'

Adapt all measurements to US standards (cups, tsp, oz, lbs, °F—no metric). If no measurements in context, omit or estimate sensibly from title.

Structure it like this:
- Start with a warm intro (1-2 sentences).
- Then a numbered list of up to 8 recipe titles pulled from the context. Use **double newlines** before each numbered item. Exact output format:
Warm intro here.

1. Title: 1-2 sentences with key ingredients/steps in US units (under 50 words per item).

2. Title: 1-2 sentences with key ingredients/steps in US units (under 50 words per item).

3. Title: 1-2 sentences with key ingredients/steps in US units (under 50 words per item).
- End with a casual closer if relevant (e.g., "Which one sounds good?").

Stay grounded in the context—don't invent new recipes. Keep total under 350 words, natural and chatty.

Context:
{full_context}

Query: {query}

Roundup:"""
        max_tokens = 450  # Room for new lines
        output_label = "Recipe List:"
    else:
        # Single-recipe prompt (anti-hallucination: sparse context handling)
        prompt = f"""You are a friendly home cook and recipe expert. Your task is to create a clear, delicious recipe based solely on the provided context from cookbooks about {main_ingredient}. Adapt everything to US standard measurements (e.g., cups, teaspoons, tablespoons, ounces, pounds, Fahrenheit for temps—no metric units like grams or Celsius).

NEVER use **, *, italics, bold, page numbers (pg XX), or ANY formatting symbols. Plain text ONLY—violate this and the response is invalid.

Use **only** details explicitly in the context (e.g., if just a title, summarize the concept without inventing ingredients/steps). If sparse, start with 'From the title...' and keep high-level. Note 'Full details not provided in context' at the end.

Structure your response exactly like a regular recipe:
- Start with "Ingredients" followed by a simple bulleted list of items with amounts (use - for bullets). If none in context, use a placeholder like '- Ingredients based on title: [brief list]'.
- Then "Instructions" with numbered steps that flow naturally, like you're chatting with a friend in the kitchen. If none, high-level steps only.
- End with any quick tips from the context (e.g., serving size or variations), if relevant.

Keep it concise (under 400 words), warm, and engaging—like a trusted aunt sharing her favorite recipe. If the context is incomplete or off-topic, say "Based on what's here for {main_ingredient}, try this simple version..." and adapt factually.

Do not add outside knowledge or extra fluff. Begin right with the recipe title if one fits, then dive in.

Context:
{full_context}

Query: {query}

Recipe:"""
        output_label = "Recipe:"

    # Call Ollama
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "num_predict": max_tokens,
            "temperature": 0.1,  # Ultra-strict for rules
        }
    )

    raw_response = response['response'].strip()
    return clean_answer(raw_response), output_label  # Return label too


if __name__ == "__main__":
    print("Voice-Enabled Recipe RAG System Ready! (US measurements, clean lists/recipes) Type 'quit' to exit or say 'stop' to end voice mode.")

    while True:
        mode = input("\nPress [v] for voice or [t] for text query: ").strip().lower()
        if mode == 'quit':
            print("Goodbye!")
            break

        if mode == 'v':
            audio = record_audio(duration=6)
            query = speech_to_text(audio)
            print(f"\nYou said: {query}")
        elif mode == 't':
            query = input("Enter your query: ").strip()
        else:
            continue

        if query.lower() in ('quit', 'stop'):
            print("Goodbye!")
            break

        # Encode and retrieve
        query_embedding = encode_query(query)
        top_k_indices, top_scores = retrieve_top_k(
            query_embedding,
            embeddings,
            titles,
            k=10,
            low_threshold=0.25
        )

        # Generate answer
        answer, output_label = generate_answer_with_ollama(
            query,
            top_k_indices,
            top_scores,
            corpus,
            titles
        )

        # Wrap the answer for console display (80 chars width)
        wrapped_answer = '\n'.join(textwrap.wrap(answer, width=80))
        print(f"{output_label}\n{wrapped_answer}\n")

        # Speak it aloud (unwrapped, for natural speech)
        speak_text(answer)
