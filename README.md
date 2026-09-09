# Multiple Choice Question Classification Engine

Sequence classification system for 5-option multiple-choice question answering and ranking using fine-tuned transformer architectures (DeBERTa-v3-small and RoBERTa-base).

## Overview

The engine supports:
- Single-instance inference with probability distribution across all 5 candidate options.
- Top-3 candidate ranking with configurable Test-Time Augmentation (TTA).
- Ensemble weighting: 0.70 DeBERTa-v3-small + 0.30 RoBERTa-base.
- Batch CSV evaluation with exportable prediction files matching submission schemas.

## Models
- **DeBERTa Model**: `CaptainRohith/smart-mcq-deberta`
- **RoBERTa Model**: `CaptainRohith/smart-mcq-roberta`

## Local Execution
```bash
pip install -r requirements.txt
streamlit run app.py
```
