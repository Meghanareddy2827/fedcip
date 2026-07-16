# FedCIP — Federated Cybersecurity Intelligence Platform

A trust-aware, standards-based federated learning platform for privacy-preserving
cyber threat intelligence (CTI) sharing across organizations.

Base reference: Mrabet, M. (2025). TrustFed-CTI: A Trust-Aware Federated Learning
Framework for Privacy-Preserving Cyber Threat Intelligence Sharing Across
Distributed Organizations. Future Internet, 17(512).

---

## Team setup (do this once per laptop)

git clone https://github.com/<your-org-or-username>/fedcip.git
cd fedcip
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Team workflow

- Never commit directly to main.
- Create a branch per feature: git checkout -b <yourname>/<feature>
- Commit small, working chunks with clear messages.
- Push your branch and open a Pull Request on GitHub for review before merging.
- Pull main regularly to stay in sync: git pull origin main

## Project structure

fedcip/
  data/              # datasets (raw data itself should NOT be committed)
  clients/           # per-organization local training code
  server/            # aggregator / coordinator code
  trust_engine/      # Q/C/R/H trust score logic
  privacy/           # differential privacy + encryption code
  dashboard/         # dashboard / API (later phase)
  notebooks/         # experiment notebooks
  requirements.txt
  .gitignore
  README.md

## Handling datasets

Do not commit large dataset files to Git. Put a download link or instructions
in data/README.md instead, and keep data/ in .gitignore.

## Current implementation phase

- [ ] Phase 1: Setup & literature refinement
- [ ] Phase 2: Data pipeline
- [ ] Phase 3: Baseline federated learning (plain FedAvg)
- [ ] Phase 4: Trust engine (Q/C/R/H)
- [ ] Phase 5: Privacy layer (DP + secure aggregation)
- [ ] Phase 6: Hierarchical aggregation + audit ledger
- [ ] Phase 7: Adversarial robustness testing
- [ ] Phase 8: Dashboard + integration
- [ ] Phase 9: Evaluation & report writing

<https://drive.google.com/drive/folders/17ZduCh4AiSS-l8JxQZPqEZ55M9cs5UDr?usp=sharing>
