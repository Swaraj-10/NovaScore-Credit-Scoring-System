NovaScore Credit Scoring System Team Grabbbb — Grab AI National Hackathon 2025, National Semifinalists
------------------------------------------------------------------------------------------------------

Overview NovaScore is an AI-driven credit scoring framework that estimates short-term default risk using behavioral signals from ride activity, transactions, and user–merchant network structure. It was developed for the Grab AI National Hackathon 2025, where the team qualified through the first three rounds and narrowly missed the finals.

Problem statement Conventional credit assessment struggles with thin-file users. The goal is to build a data-driven risk model over a 90-day observation window that captures:

trip-level engagement and reliability
transaction patterns and balances
network effects via user–merchant relationships
Key features

90-day dynamic window with weekly buckets (about 13 time steps)
Tabular aggregations for trips, cancellations, tips, earnings, transaction stats
Temporal sequences modeled with a TCN over weekly aggregates
User–merchant bipartite graph embeddings via Node2Vec (64-dim)
FT-Transformer for tabular features with per-feature tokenization
Hybrid fusion head that combines FT-Transformer, TCN, and graph vectors
LightGBM baseline for comparison and quick explainability
Probability calibration to a 300–940 NovaScore with clear decision bands
Simple fairness audit using delta TPR across city groups
Input data The pipeline expects two parquet files uploaded at runtime:

merged_trip_details_0.parquet
transaction-0.parquet These are not included in the repo. Place them in data/raw if you want local runs, or upload via the Colab file picker when prompted.
How it works

windowing: computes an anchor end date from the latest timestamp and selects the last 90 days
labeling: marks a user positive if any transaction status indicates default/chargeback-like behavior
tabular features: per-user sums, means, counts, cancel/incident rates, payment mix, route mix, device/merchant mix
weekly sequences: per-user weekly aggregates for trips, distance, duration, cancellations, rating, earnings, spend, transactions, unique merchants; normalized across the window
graph embeddings: builds a user–merchant bipartite graph and learns Node2Vec embeddings; each user gets a dense 64-d vector
baseline model: LightGBM trains on tabular features and reports AUROC
hybrid model: FT-Transformer encodes tabular features, TCN encodes sequences, Node2Vec encodes graph; the vectors are concatenated and passed through a dense head
calibration: converts predicted PD to NovaScore using a logit mapping anchored at chosen PD–score pairs
output: writes predictions_90d_full.csv with columns [user_id, y_true, pd90, novascore, decision_band, group] and also saves pred_prob_90d_full.npy and score_90d_full.npy
Scoring policy

score ≥ 800: auto-approve, large limit
700–799: standard approve, medium limit
600–699: manual review, small limit
< 600: decline with repayment coaching
Quick start (Colab or local) Colab

open the notebook or a Python cell in Colab
install requirements for the session and run the script
when prompted, upload merged_trip_details_0.parquet and transaction-0.parquet
Local

python -m venv .venv && source .venv/bin/activate (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
run your script or notebook and point to the parquet files in data/raw
Evaluation

primary metric: AUROC
reference: LightGBM baseline vs hybrid model
fairness check: delta TPR across city groups
Notes and assumptions

the script is designed to be resilient to slight schema variations and type coercions
graph embeddings are zero-vectors for users unseen in the random walks
calibration anchors are adjustable; defaults are chosen for a balanced score spread
no real user data is included in this repository
Credits Team Grabbbb Project codename: NovaScore Credit Scoring System Originally developed for the Grab AI National Hackathon 2025 (national semifinalists)
Youtube Video:-https://youtu.be/tunNvJc6hN0

License MIT License
