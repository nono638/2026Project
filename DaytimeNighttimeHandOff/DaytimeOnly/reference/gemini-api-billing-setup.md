# Gemini API: Upgrading from Free Tier to Paid

You already have a `GOOGLE_API_KEY` in your `.env` file that works with AI Studio.
The key itself doesn't change — you just need to link billing to the Google Cloud
project behind that key. Here's how.

## What You Need

- Your existing Google account (the one with the $20/month AI Pro subscription)
- A credit card (for billing — you'll pay per API call, pennies for our usage)
- ~5 minutes

## Steps

### 1. Open Google AI Studio

Go to: https://aistudio.google.com

Sign in with the same Google account you used to create your API key.

### 2. Enable billing on your project

- Click the **gear icon** (Settings) in the left sidebar
- Look for **"Billing"** or **"Upgrade to Paid Tier"**
- AI Studio lets you link a billing account directly — you do NOT need to go to
  the Google Cloud Console separately
- Follow the prompts to add a credit card

Alternatively, go directly to: https://aistudio.google.com/apikey
- Find your existing API key
- There should be an option to upgrade the associated project to paid billing

### 3. Verify it worked

Once billing is linked, your existing `GOOGLE_API_KEY` automatically gets Paid Tier 1
rate limits. Nothing changes in your `.env` file — same key, higher limits.

**Free tier limits (what you have now):**
| Model | Requests/min | Requests/day |
|---|---|---|
| Gemini 2.5 Pro | 5 | 100 |
| Gemini 2.5 Flash | 10 | 250 |
| Flash-Lite | 15 | 1,000 |

**Paid Tier 1 limits (after linking billing):**
Higher RPM and RPD across all models — enough for our 50-example experiments.

### 4. Set a spending cap (important!)

After enabling billing, set a budget alert in Google Cloud Console so you don't
get surprised:
- Go to: https://console.cloud.google.com/billing
- Find your billing account → **Budgets & alerts**
- Set a $5 or $10 monthly budget with email alerts

### 5. Test it

Run experiment 0 again to verify Gemini Pro scores all 50 examples:
```
python scripts/run_experiment_0.py --skip-generation
```
(`--skip-generation` reuses the existing RAG answers and just re-scores them)

## Cost Estimate

For 50 examples × 3 Gemini judges × 1 API call each = 150 calls.
At ~$0.001/call for Pro and less for Flash/Flash-Lite, total cost: **well under $1**.

## What You Do NOT Need

- A new API key (your existing `GOOGLE_API_KEY` stays the same)
- Google Cloud SDK or CLI
- Any code changes — the experiment script already handles all Gemini models

## Your $20/month AI Pro Subscription

That subscription is for the consumer Gemini app (gemini.google.com). It's completely
separate from the API. It does NOT affect API rate limits. You're paying for two
different things — the subscription for chat, and (now) pay-per-call for the API.

## Sources

- Billing setup: https://ai.google.dev/gemini-api/docs/billing
- Rate limits by tier: https://ai.google.dev/gemini-api/docs/rate-limits
- API key management: https://ai.google.dev/gemini-api/docs/api-key
- Pricing: https://ai.google.dev/gemini-api/docs/pricing
