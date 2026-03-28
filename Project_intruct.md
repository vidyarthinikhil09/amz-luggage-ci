# MOONSHOT AI Agent Internship Assignment

## Build a competitive intelligence dashboard for luggage brands on Amazon India

This assignment tests structured data collection, review synthesis, product thinking, and UI execution. The goal is to turn messy marketplace signals into a decision-ready dashboard.

---

## Core workflow
Scrape - Analyze - Compare - Present

---

## Primary data source
Amazon India product listings and reviews

---

## Minimum brand coverage
4+ luggage brands

---

## What we care about
- Insight quality
- Agentic thinking
- UI polish

---

## Submission package
- Working dashboard  
- Source code  
- README  
- Cleaned dataset  

**Optional but recommended:**
- A short Loom walkthrough explaining your architecture, tradeoffs, and key insights  

---

# Assignment Objective

Build an interactive dashboard that scrapes and synthesizes customer reviews and pricing data for selected luggage brands selling on Amazon India.

---

## What the dashboard should answer

- Which brands are priced at a premium versus value-focused price bands?
- Which brands rely on higher discounting to drive demand?
- What customers consistently praise or complain about?
- Which brands appear to win on sentiment relative to price?

---

# Core Requirements

## 1. Sentiment Analysis
- Scrape customer reviews for products across selected brands.
- Generate an overall sentiment score for each brand.
- Surface major positive and negative themes.
- Highlight recurring complaints and recurring praise.

---

## 2. Pricing Insights
- Show average selling price by brand.
- Show average listed discount by brand.
- Show product-level price spread where useful.
- Make premium versus mass-market positioning obvious.

---

## 3. Competitive Analysis
- Compare multiple brands side by side.
- Benchmark:
  - Price
  - Discount
  - Rating
  - Review count
  - Sentiment
- Help a user quickly identify who is winning and why.

---

## 4. Interactive UI
- Dashboard must be clean, intuitive, and visually strong.
- Use click-based interactions, filters, and drilldowns.
- Charts, tables, and summaries should update dynamically.

> Important: This should NOT be a static report disguised as a dashboard.

---

# Minimum Data Scope

Suggested starting brands:
- Safari
- Skybags
- American Tourister
- VIP
- Aristocrat
- Nasher Miles

You may choose your own brand set if it improves analysis.

| Requirement | Minimum |
|------------|--------|
| Brands | 4+ |
| Products per brand | 10+ |
| Reviews per brand | 50+ |

> If data availability varies, document limitations clearly.

---

# Expected Dashboard Views

## Dashboard Overview
- Total brands tracked
- Total products analyzed
- Total reviews analyzed
- Average sentiment snapshot
- Pricing overview

---

## Brand Comparison View
- Average price
- Average discount %
- Average star rating
- Review count
- Sentiment score
- Top pros and cons

---

## Product Drilldown
- Product title
- Price
- List price
- Discount
- Rating
- Review count
- Review synthesis
- Top complaint themes
- Top appreciation themes

---

## Filters and Interactions
- Brand selector
- Price range filter
- Minimum rating filter
- Luggage category/size filter
- Sentiment filter
- Sortable comparison tables

---

# Technical Expectations

You may choose your own stack.

Focus areas:
- Reproducible scraping workflow
- Clean dataset structure
- Documented sentiment methodology
- End-to-end working dashboard

### Suggested Tech Stack
- Python
- Playwright / Selenium
- Pandas
- Streamlit / React
- Plotly / Chart libraries
- LLM or sentiment model

---

# Deliverables

## Required
- Working dashboard
- Source code
- README (setup + approach)
- Cleaned dataset

---

## Recommended
- 3–5 minute Loom/video walkthrough
- Architecture diagram
- Notes on limitations and improvements

---

# Evaluation Rubric

| Criteria | Description | Score |
|--------|------------|------|
| Data collection quality | Correct, structured, usable data | 20 |
| Analytical depth | Sentiment logic, themes, reasoning | 20 |
| Dashboard UX/UI | Layout, hierarchy, interactions | 20 |
| Competitive intelligence | Meaningful comparisons, insights | 15 |
| Technical execution | Code quality, architecture, docs | 15 |
| Product thinking | Decision-making relevance | 10 |

**Total: 100**

---

# Bonus Points

- Aspect-level sentiment (wheels, handle, material, zipper, size, durability)
- Anomaly detection (e.g., durability complaints despite high ratings)
- Value-for-money analysis (sentiment adjusted by price)
- Review trust signals (fake/repeated patterns)
- Agent Insights:
  - Auto-generate 5 non-obvious conclusions

---

# What an Excellent Submission Looks Like

- Goes beyond ratings and discounts
- Explains *why* a brand is winning
- Converts reviews into structured insights
- Includes an **Agent Insights layer** with deep conclusions

---

# Submission Notes

- Timeline: 5–7 days
- Submit:
  - GitHub repo
  - Hosted app link (if available)
  - README
  - Dataset
  - Optional video walkthrough

---

## Final Expectation

A strong submission:
- Uses scraped + synthesized data
- Explains:
  - Why a brand is winning
  - Where value is strongest
  - What decisions should be made next