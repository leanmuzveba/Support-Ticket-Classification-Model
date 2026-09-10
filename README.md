# Future Interns — Machine Learning Task 2: Support Ticket Classification

## About This Project

This is my submission for Future Interns' ML Track Task 2. The brief was to build a
system that automatically classifies customer support tickets by category and assigns
them a priority level, to help support teams triage and respond faster.

## Dataset

I used a real, anonymized support ticket dataset (originally Microsoft's internal IT
helpdesk data, ~48,500 tickets) sourced from Kaggle. Each ticket has a title, a body of
free text describing the issue, a ticket type (Request or Incident), a category code,
and an urgency level.

**Note on the labels**: the category and urgency values in this dataset are anonymized
integer codes rather than readable names — the provider stripped the real category
names before publishing it. Rather than invent fake category labels that would
misrepresent the data, I kept them as "Category 1," "Category 2," etc. In a real
deployment, these would map to an actual company's ticket taxonomy.

### A note on how I chose this dataset

My first choice was a different, more polished-looking Kaggle dataset with readable
category names (Refund request, Technical issue, etc.) and a clean priority column.
Before building anything on it, I tested whether the ticket text actually predicted
those labels — and it didn't. I found the dataset only had 16 unique subject lines
reused across all 8,469 tickets, spread almost evenly across every category, and a
model trained on it scored at exactly chance level (about 20% on a 5-class problem).
The text and the labels had been generated independently of each other.

I mention this because it was a genuinely useful lesson: a dataset can look perfect on
the surface (right columns, balanced classes, real-sounding labels) and still be
useless for the actual task. I switched to the dataset described above after verifying
it had real, learnable signal.

## Approach

1. **Text cleaning** — combined the ticket title and body, lowercased everything, and
   stripped non-alphabetic characters.
2. **Handling label structure** — discovered that urgency only varies meaningfully for
   "Request"-type tickets (~14,000 of the 48,500); the "Incident"-type tickets all
   share a single urgency value, which would have been fake signal to train on. So the
   priority model is trained only on the Request subset, with its 3 real urgency
   levels mapped directly to Low/Medium/High.
3. **Vectorization** — TF-IDF (up to 5,000 features, English stop words removed).
4. **Modeling** — Logistic Regression with balanced class weights (the category
   distribution is heavily skewed — one category alone makes up ~70% of tickets — so
   balanced weighting matters for the model to actually learn the smaller categories
   rather than just always guessing the biggest one).
5. **Evaluation** — accuracy, macro F1 (more informative than accuracy alone given the
   class imbalance), full classification report, and a confusion matrix for each model.

## Results

| Model | Accuracy | Majority-Class Baseline | Macro F1 |
|---|---|---|---|
| Category classifier (13 classes) | 78.97% | 70.15% | 41.35% |
| Priority classifier (Low/Medium/High) | 57.25% | 48.46% | 54.87% |

Both models beat their baseline meaningfully, confirming the text carries real
predictive signal. The category model's macro F1 is noticeably lower than its accuracy
— that's because a handful of categories have very few examples (some under 15 in the
test set), and the model struggles on those rare cases even though it does well on the
common ones. That's a real limitation worth being upfront about, not a reason to hide
the macro F1 number.

## What the System Does

Given a new ticket's title and description, the system predicts which category it
belongs to, and — for request-type tickets — how urgent it is (Low/Medium/High).

**For a support team**, this means incoming tickets can be automatically routed to the
right queue and flagged by urgency the moment they arrive, instead of a human reading
and triaging each one manually. High-priority requests surface immediately rather than
waiting in a general queue; misrouted tickets (sent to the wrong specialist team) drop
significantly, since the category prediction beats guesswork by a wide margin (79% vs.
70% baseline).

## Limitations

- Category and priority labels are anonymized codes, not real category names — this
  demonstrates the technique, but a production version would need the actual business's
  category taxonomy.
- The category model performs much better on common categories than rare ones; rare
  categories would need more training examples or a different approach (e.g. combining
  small categories, or collecting more data) before this could be trusted in production.
- Priority prediction only applies to "Request"-type tickets in this dataset, since
  "Incident"-type tickets didn't have varying urgency labels to learn from.
- This is trained on IT helpdesk tickets specifically — applying it to a different kind
  of support team (e.g. e-commerce customer service) would need retraining on that
  team's own ticket data.

## Tools Used

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook.

## Files in This Repo

- `notebooks/02_ticket_classification.ipynb` — full analysis: data verification, text
  cleaning, both classifiers, evaluation
- `data/all_tickets.csv` — source dataset