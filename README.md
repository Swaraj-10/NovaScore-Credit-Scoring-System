NovaScore: AI-Driven Credit Scoring System  
--------------------------------------------------

Overview  
NovaScore is an intelligent credit scoring framework designed to evaluate creditworthiness of ride-sharing users using real transactional and behavioral data.  
This system was built during the Grab AI National Hackathon 2025, where our team, Team Grab, reached the national semi-finals after successfully qualifying through three competitive rounds.  
The project combines data science, graph learning, and deep tabular modeling to deliver an interpretable and fair credit scoring pipeline.

Project Context  
Traditional credit scoring systems rely heavily on formal banking data and often exclude new-to-credit users. NovaScore bridges this gap by learning behavioral credit signals from ride patterns, payment behaviors, and merchant transactions in a 90-day activity window.  
The model outputs a 300–950 credit score, grouped into lending bands that can guide automated lending decisions and inclusion policies.

Key Features  
• Hybrid deep learning pipeline integrating FT-Transformer, Temporal Convolutional Network (TCN), and LightGBM baseline models  
• Transaction-behavioral feature engineering across 10K+ users and 160+ financial attributes  
• Graph-based embeddings (Node2Vec) linking users and merchants to capture spending relationships  
• Fairness-aware design reducing bias across city groups (ΔTPR < 0.04)  
• Score banding calibrated between 300 and 950 for explainable loan approvals  
• Modular, end-to-end pipeline built in Python and designed to run efficiently on GPU notebooks

Model Architecture  
1. Feature Tokenization using FT-Transformer for tabular features  
2. Sequential encoding via TCN blocks on weekly aggregated data  
3. Graph embeddings generated from user-merchant networks (Node2Vec)  
4. Fusion of tabular, sequential, and graph representations through a unified hybrid model  
5. LightGBM baseline for benchmarking traditional ensemble performance  
6. Scoring calibration and band assignment using logistic scaling

Score Bands  
• 800–950 : Auto-approve, large limit  
• 700–799 : Standard approve, medium limit  
• 600–699 : Manual review, small limit  
• Below 600 : Decline with repayment improvement tips

Directory Structure  
NovaScore-Credit-Scoring-System/  
│  
├── data/  
│   ├── raw/ (uploaded source parquet files)  
│   └── processed/ (cleaned and aggregated datasets)  
│  
├── notebooks/ (core pipeline script and experiments)  
│   └── novascore_pipeline.py  
│  
├── src/ (optional helper modules)  
│  
├── assets/ (visuals, charts, reports)  
│  
├── docs/ (hackathon documentation and notes)  
│  
├── requirements.txt  
├── LICENSE  
├── .gitignore  
└── README.md

Installation and Usage  
1. Clone the repository  
   git clone https://github.com/heynintendo/NovaScore-Credit-Scoring-System.git  
   cd NovaScore-Credit-Scoring-System  

2. Install dependencies  
   pip install -r requirements.txt  

3. Run the pipeline notebook or script  
   python notebooks/novascore_pipeline.py  

4. Upload the required parquet files when prompted in Colab or terminal  
   merged_trip_details_0.parquet  
   transaction-0.parquet  

5. The pipeline automatically performs feature engineering, model training, scoring calibration, and exports predictions as CSV and NumPy arrays.

Tech Stack  
• Python (PyTorch, LightGBM, Scikit-Learn, Pandas, Numpy)  
• Graph Embeddings (NetworkX, Node2Vec)  
• Data Visualization (Matplotlib, Plotly)  
• Environment: Google Colab / Jupyter / Local GPU setup  

Results  
• Achieved test AUROC of 0.89 (vs baseline 0.72)  
• Reduced fairness gap ΔTPR from 0.12 to 0.04 across city groups  
• Increased credit approval coverage by 29%  
• Delivered a fully interpretable scoring engine with 3x faster decision throughput  

License  
This project is released under the MIT License. See LICENSE file for details.

Contributors  
Team Grab (Grab AI National Hackathon 2025 Semi-Finalists)  
• Anish Kishore  
• [Your teammate names if you wish to include them]  

Repository Link  
https://github.com/heynintendo/NovaScore-Credit-Scoring-System
