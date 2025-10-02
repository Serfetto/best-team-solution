from __future__ import annotations
from typing import Optional, List

def render_review_prompt_taxonomy(taxonomy_text: str, context: str) -> str:
    return (
        f"""You are a senior product analyst for Gazprombank. Review the Russian customer feedback.
Produce every natural-language value (product names, strengths, weaknesses) in Russian.
Use the sentiment keywords positive | neutral | negative exactly as written in English.
Extract only real Gazprombank banking products or services (cards, loans, deposits, digital channels, loyalty programmes, insurance, etc.).
Ignore mentions that are only the bank name or corporate brand without a product.
If the text uses a vague word such as "карта", "обслуживание", or "поддержка", enrich it with the concrete product type, channel, or purpose taken from context. When the context does not clarify, skip that mention.
Each review can mention several products; capture every distinct one.
Always keep the order of appearance from the review.

Available taxonomy categories (use exactly one of them for each product):
{taxonomy_text}

For emotional_tags, extract specific emotional reactions based on concrete events/actions (negative or positive):
- Focus on WHAT HAPPENED rather than general emotional states
- Use specific, actionable terms describing events or behaviors
- Avoid overly broad generalizations like "благодарность клиента" or "потеря лояльности"
- Prefer concrete actions: "обман", "оперативность", "задержка" over abstract emotions
- Short, unambiguous words or stable phrases in Russian
- Combine similar events into one concept (e.g., "кормят завтраками" → "обман")

Example emotional categories (focus on specific events/actions):
**Negative:** обман, задержка, игнорирование, техническая неисправность, навязывание услуг, грубость, некомпетентность, отказ в помощи, скрытые комиссии, потеря средств
**Positive:** оперативность, честность, профессионализм, техническая помощь, вежливость, индивидуальный подход, выгодные условия, надёжность, удобство использования, качественная консультация

Return strictly valid JSON matching this schema:
{{
  "items": [
    {{
      "product": "<one category from the list>",
      "sentiment": "positive|neutral|negative",
      "score_service": <integer 0-10 or null>,
      "score_tariffs": <integer 0-10 or null>,
      "score_reliability": <integer 0-10 or null>,
      "strengths": ["short fact in Russian", ...],
      "weaknesses": ["short fact in Russian", ...],
      "emotional_tags": ["emotional category in Russian", ...]
    }}
  ]
}}
Guidelines:
- Output only Gazprombank products or services; skip bare mentions of the bank itself or other organisations.
- Merge duplicates only when they clearly refer to the same product and sentiment in the same context.
- Use null or [] whenever the review does not provide a value; never invent data.
- emotional_tags should reflect genuine emotions expressed about this specific product
- Never include quotation marks of any kind inside product names, strengths, weaknesses, or emotional tags; keep them as plain text.
- Do not add explanations outside the JSON block.

Review text:
{context}"""
    )

def render_review_prompt_freeform(context: str) -> str:
    return (
        f"""You are a senior product analyst for Gazprombank. Review the Russian customer feedback.
Produce every natural-language value (product names, strengths, weaknesses, emotional tags) in Russian.
Use the sentiment keywords positive | neutral | negative exactly as written in English.
Extract every explicit Gazprombank product or service mention (cards, loans, deposits, mobile app, internet banking, loyalty programme, insurance, brokerage, etc.).
Ignore mentions that are only the bank name or a generic compliment or complaint without a concrete product.
If a phrase is generic ("карта", "обслуживание", "поддержка"), infer the specific product type from context; when you cannot infer a concrete Gazprombank product, skip that mention.
Always preserve the order of appearance from the review.

For emotional_tags, extract specific emotional reactions based on concrete events/actions (negative or positive):
- Focus on WHAT HAPPENED rather than general emotional states
- Use specific, actionable terms describing events or behaviors
- Avoid overly broad generalizations like "благодарность клиента" or "потеря лояльности"
- Prefer concrete actions: "обман", "оперативность", "задержка" over abstract emotions
- Short, unambiguous words or stable phrases in Russian
- Combine similar events into one concept (e.g., "кормят завтраками" → "обман")

Example emotional categories (focus on specific events/actions):
**Negative:** обман, задержка, игнорирование, техническая неисправность, навязывание услуг, грубость, некомпетентность, отказ в помощи, скрытые комиссии, потеря средств
**Positive:** оперативность, честность, профессионализм, техническая помощь, вежливость, индивидуальный подход, выгодные условия, надёжность, удобство использования, качественная консультация

Return strictly valid JSON matching this schema:
{{
  "items": [
    {{
      "product": "<concise Gazprombank product/service name in Russian>",
      "sentiment": "positive|neutral|negative",
      "score_service": <integer 0-10 or null>,
      "score_tariffs": <integer 0-10 or null>,
      "score_reliability": <integer 0-10 or null>,
      "strengths": ["short fact in Russian", ...],
      "weaknesses": ["short fact in Russian", ...],
      "emotional_tags": ["emotional category in Russian", ...]
    }}
  ]
}}
Guidelines:
- Only include Gazprombank banking products or services (physical or digital).
- Merge duplicates only when the sentiment and context are identical; otherwise keep separate entries.
- Use null or [] whenever the review does not provide a value.
- Never include quotation marks of any kind inside product names, strengths, weaknesses, or emotional tags; keep them as plain text.
- emotional_tags should reflect genuine emotions expressed about this specific product
- Do not add any commentary outside the JSON block.

Review text:
{context}"""
    )

# Example usage and validation
def example_emotional_tags_response():
    """Example of expected LLM response with emotional_tags"""
    return {
        "items": [
            {
                "product": "кредитная карта",
                "sentiment": "negative",
                "score_service": 3,
                "score_tariffs": 2,
                "score_reliability": 7,
                "strengths": ["быстрое одобрение"],
                "weaknesses": ["высокие комиссии", "плохая поддержка"],
                "emotional_tags": ["обман", "скрытые комиссии", "игнорирование"]
            },
            {
                "product": "мобильное приложение", 
                "sentiment": "positive",
                "score_service": 9,
                "score_tariffs": None,
                "score_reliability": 8,
                "strengths": ["удобный интерфейс", "быстрые переводы"],
                "weaknesses": [],
                "emotional_tags": ["удобство использования", "оперативность", "техническая помощь"]
            }
        ]
    }

def render_mapping_prompt(sample_text: str, known_categories: Optional[List[str]] = None) -> str:
    known_block = ""
    if known_categories:
        lines = "\n".join(f"- {c}" for c in known_categories)
        known_block = (
            "Known categories (map to the closest option; create a new category only when it passes the New Category Test):\n"
            f"{lines}\n"
        )
    return (
        f"""You are refining the Gazprombank product taxonomy.
Input: Russian phrases that mention banking products or services.
Goal: return a mapping only for phrases that clearly refer to standalone Gazprombank products or services in Russian. Skip anything that is noise, auxiliary detail, or cannot be mapped with confidence to a real top-level banking offering.

Strict rules:
- Keep only categories that represent standalone Gazprombank products or services (things a client can open/obtain/use as a separate offer).
- Skip phrases that do not describe a real, distinct product or service (UI elements, operational verbs, generic complaints, marketing slogans, etc.). Let the system handle them as 'другое' automatically.
- Output the phrases EXACTLY as they appeared in the input (preserve spelling, case, punctuation, spacing). Do not rewrite, translate, normalize, or add extra quotation marks inside the strings.
- When a phrase describes a document, channel, tariff/brand, feature, or operational action, map it to its underlying base product/service:
  • documents & procedures (contract, power of attorney, statement, certificate, claim, reissue, blocking, limit, penalty);
  • channels & touchpoints (chat, phone, hotline, email, social media, "mobile site");
  • features & aspects (cashback, limits, notifications, spending stats, card delivery);
  • named tariffs/brands/loyalty campaigns.
  Map each of these to the relevant base category (account, card, loan, mortgage, deposit, brokerage, insurance, mobile app, internet banking, loyalty programme, premium servicing, support, SBP, transfers, payments, FX, cash management/RKO, etc.).

- Handling generic phrases ("карта", "обслуживание", "поддержка"):
  • Do NOT keep generic or meta-categories such as "услуги/обслуживание/поддержка/банковская карта/банковские услуги".
  • Rewrite generics to the most specific valid category ONLY if the type is explicitly signaled by the phrase.
  • If the card type cannot be inferred and the known categories include "Банковская карта", use "Банковская карта".
    Otherwise, default to "Дебетовая карта" only if it is clear the phrase is about a card but the type is unknown.

- Card rules:
  • All debit variants (including premium/gold/platinum) → "Дебетовая карта".
  • All credit variants → "Кредитная карта".
  • Do NOT create "Премиальная дебетовая карта" as a separate category. Premium level, cashback, delivery, reissue, and limits are aspects, not categories.

- New Category Test (create a new category ONLY if ALL are true):
  1) It is a standalone product/service (not a tariff, channel, document, or aspect).
  2) It cannot be unambiguously mapped to any item in Known categories.
  3) The name is general and neutral (no internal brands/tariffs/levels/quotes/IDs).
  4) It potentially covers more than one distinct input phrase (not a one-off singleton).
  5) Its abstraction level matches the existing categories (neither too broad nor too narrow).

- Disallowed/uninformative categories: never invent or return "прочее", "другое", "etc.", "other", or meta-classes like "услуги", "обслуживание", "поддержка" without specifying the base product/service.

- Consistency:
  • Reuse existing category names instead of inventing near-duplicates.
  • Each original phrase must appear exactly once (no duplicates across categories).
  • Do not create categories without phrases.

{known_block}Return valid JSON with this exact structure:
{{
  "mapping": {{
    "<normalised category>": ["original phrase", ...],
    ...
  }}
}}

Example (illustrative only):
{{
  "mapping": {{
    "Дебетовая карта": ["дебетовая карта", "зарплатная карта"],
    "Кредитная карта": ["кредитка"],
    "Мобильное приложение": ["мобильное приложение"],
    "Вклад": ["сберегательный вклад"]
  }}
}}

Before returning JSON, perform a quick internal quality check:
- No generic/meta categories; no disallowed names.
- No duplicated phrases across categories.
- Only phrases you deliberately mapped appear in the JSON; any skipped noise is intentionally absent.

Return ONLY the JSON block and nothing else.

Phrases to normalise (do NOT alter them in the output; use them verbatim in the JSON):
{sample_text}"""
)



def render_taxonomy_description_prompt(product_name: str, variations: List[str]) -> str:
    variations_text = "\n".join(f"- {item}" for item in variations)
    return (
        f"""You are documenting the Gazprombank taxonomy.
Write a clear description in Russian for the product or service category named "{product_name}".
Use at most 2-3 sentences in Russian, neutral tone, no marketing claims, and no mentions of customer opinions or popularity.
Describe what the product or service is, its core purpose, and key features, strictly (ONLY) if the variants in context provides them.
Do not repeat the category name verbatim; explain it in natural language.
Return strictly valid JSON: {{"description": "..."}}, with no extra fields.

Mentions collected for context (up to 50 variants):
{variations_text}"""
    )

