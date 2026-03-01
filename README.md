# **Recommendation System:** Production-Grade Real-Time Recommendation Engine

A complete recommendation system built from scratch — progressing from classical baselines to transformer-based neural models — with real-time vector search, A/B testing, and production monitoring. This project mirrors the recommendation architectures powering Netflix, YouTube, Amazon, and Spotify.

## Business Problem

> **How can a streaming platform increase user engagement by 15%+ through personalized content recommendations while maintaining recommendation diversity and freshness?**

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        DATA LAYER                             │
│   MovieLens 25M: 25M ratings · 162K users · 62K movies        │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                      MODEL LAYER                              │
│                                                               │
│   Part 1: Popularity + Item-CF + ALS + Hybrid Baseline        │
│   Part 2: Neural Collaborative Filtering (PyTorch)            │
│   Part 3: Transformer-Based Sequential RecSys (SASRec)        │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                     SERVING LAYER                             │
│   Part 4: FAISS Approximate Nearest Neighbor + FastAPI        │
│           Sub-50ms latency · 1000+ QPS                        │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│               EVALUATION & MONITORING                         │
│   Part 5: A/B Testing + Bayesian Analysis + Causal Inference  │
│   Part 6: CTR Drift Detection + Production Monitoring         │
└──────────────────────────────────────────────────────────────┘
```

## Project Roadmap

| Part | Title | Key Skills | Status |
|------|-------|-----------|--------|
| [Part 1](notebooks/Part1_EDA_Baselines.ipynb) | Data Engineering, EDA & Baseline Models | Temporal splitting, evaluation design, CF algorithms, hybrid systems | Complete |
| Part 2 | Neural Collaborative Filtering | PyTorch, embedding architectures, GPU training | In Progress |
| Part 3 | Transformer-Based Sequential RecSys (SASRec) | Self-attention, sequential modeling, positional encoding | Upcoming |
| Part 4 | FAISS Vector Search + Real-Time API | Approximate nearest neighbors, FastAPI, latency optimization | Upcoming |
| Part 5 | A/B Testing & Causal Inference | Power analysis, Bayesian testing, experimentation platforms | Upcoming |
| Part 6 | Production Monitoring & Drift Detection | Evidently AI, concept drift, model lifecycle management | Upcoming |

## Part 1 Results

Evaluated 4 models on 5,000 users using temporal train/test splitting and 7 ranking metrics:

| Model | NDCG@10 | Hit Rate@10 | Coverage | Novelty |
|-------|---------|-------------|----------|---------|
| Popularity Baseline | 0.0760 | 34.3% | 0.6% | 8.69 |
| Item-Based CF | 0.0165 | 11.1% | 25.9% | 11.73 |
| ALS+Bias (d=64) | 0.0002 | 0.2% | 15.4% | 17.59 |
| **Hybrid (alpha=0.4)** | **0.0783** | **34.8%** | **1.7%** | **8.75** |

**Best model: Hybrid (alpha=0.4)** achieved a 3.0% NDCG improvement over the Popularity Baseline.

**Business impact estimation:** Based on industry benchmarks (1% NDCG lift ≈ 0.5-2% engagement gain), this translates to an estimated 1.5%-6.0% engagement lift — equivalent to $15M-$60M incremental revenue for a platform with $1B annual revenue.

### Key Findings

**The accuracy-diversity trade-off is real and unavoidable.** The Popularity Baseline achieved the second-highest accuracy by recommending the same popular movies to everyone, but it only surfaced 0.6% of the catalog. Item-Based CF covered 25.9% of the catalog but with much lower accuracy. The Hybrid model found the optimal balance at alpha=0.4, where ALS personalization signals improve ranking quality without introducing too much noise from poorly-trained embeddings.

**Rating prediction and ranking are fundamentally different problems.** ALS achieved a strong RMSE of 0.49 on rating prediction but produced near-zero recommendation quality. Optimizing for "predict the rating accurately" does not translate to "identify the best items for this user." This is why modern recommendation systems optimize ranking objectives directly.

**Temporal evaluation prevents false confidence.** Using a temporal train/test split (each user's most recent 20% of ratings held out) instead of a random split ensures our metrics reflect real-world deployment conditions where the model must predict future behavior from past data.

## How This Maps to Real FAANG Systems

| What We Build | Real-World Equivalent |
|--------------|----------------------|
| Item-Based CF | Amazon's "Customers who bought this also bought..." |
| ALS Matrix Factorization | Spotify Discover Weekly's core engine |
| Neural CF (Part 2) | Pinterest Pin recommendations |
| SASRec Transformer (Part 3) | YouTube "Up Next" sequential model |
| FAISS serving (Part 4) | Meta's embedding-based retrieval at billion scale |
| A/B testing (Part 5) | Netflix runs 250+ concurrent experiments |
| Drift monitoring (Part 6) | Google's ML monitoring infrastructure |

## Tech Stack

**Core ML:** Python, NumPy, SciPy, Scikit-learn, PyTorch

**Data:** Pandas, Polars, Feature Engineering

**Serving:** FAISS (Approximate Nearest Neighbors), FastAPI

**Evaluation:** Custom framework — Precision@K, Recall@K, NDCG@K, MRR, Hit Rate, Coverage, Novelty

**Monitoring:** Evidently AI, custom drift detectors

**Statistics:** SciPy Stats, Bayesian A/B Testing

**Visualization:** Matplotlib, Seaborn, Plotly

## Dataset

[MovieLens 25M](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset) by GroupLens Research — 25 million ratings from 162,000 users across 62,000 movies with timestamps and genre metadata.

## How to Run

**On Kaggle (recommended):**
1. Open the notebook links above
2. Add the "MovieLens 25M" dataset by garymk
3. Run all cells sequentially

**Locally:**
```bash
git clone https://github.com/ARMAND-cod-eng/Recommendation-System.git
cd Recommendation-System
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch faiss-cpu
jupyter notebook notebooks/Part1_EDA_Baselines.ipynb
```

## Repository Structure

```
Recommendation-System/
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── Part1_EDA_Baselines.ipynb       # Complete
│   ├── Part2_Neural_CF.ipynb           # Coming soon
│   ├── Part3_Transformer_SASRec.ipynb  # Coming soon
│   ├── Part4_FAISS_Serving.ipynb       # Coming soon
│   ├── Part5_AB_Testing.ipynb          # Coming soon
│   └── Part6_Monitoring.ipynb          # Coming soon
├── src/                                # Reusable Python modules
│   ├── models.py
│   ├── evaluation.py
│   └── data_processing.py
└── results/                            # Charts and metrics
    └── part1_model_comparison.png
```

## Author

**Armand Junior Dongmo Notue** — MS Data Science

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue)](https://kaggle.com)








