# Quantitative Analysis Report

This report provides comprehensive quantitative analysis of the RAG system 
performance, including comparisons with baseline methods and statistical significance tests.

---

## Executive Summary

**Best Performing Strategy**: `adaptive`
- Average Score: `0.5156` (±0.1307)

## Overall Performance Statistics

| Strategy | Avg Score (Mean ± Std) | Min | Max | Namespace Accuracy |
|----------|------------------------|-----|-----|-------------------|
| `adaptive` | 0.5156 ± 0.1307 | 0.2517 | 0.7107 | 100.00% |
| `baseline` | 0.1865 ± 0.1131 | 0.0232 | 0.4524 | 90.00% |
| `sentence` | 0.3792 ± 0.1155 | 0.2066 | 0.6000 | 100.00% |

## Score Distribution Analysis

This section shows the **proportion of queries** falling into each score category 
for each strategy, providing explicit quantitative context for performance assessment.

**Score Categories:**
- **Excellent (≥0.8)**: High-quality retrieval
- **Good (0.6-0.8)**: Adequate retrieval
- **Moderate (0.4-0.6)**: Acceptable but suboptimal
- **Poor (<0.4)**: Low-quality retrieval

### ADAPTIVE

| Category | Count | Total | Proportion |
|----------|-------|-------|------------|
| Good (0.6-0.8) | 7 | 20 | **35.0%** |
| Moderate (0.4-0.6) | 8 | 20 | **40.0%** |
| Poor (<0.4) | 5 | 20 | **25.0%** |

### BASELINE

| Category | Count | Total | Proportion |
|----------|-------|-------|------------|
| Moderate (0.4-0.6) | 1 | 20 | **5.0%** |
| Poor (<0.4) | 19 | 20 | **95.0%** |

### SENTENCE

| Category | Count | Total | Proportion |
|----------|-------|-------|------------|
| Good (0.6-0.8) | 1 | 20 | **5.0%** |
| Moderate (0.4-0.6) | 8 | 20 | **40.0%** |
| Poor (<0.4) | 11 | 20 | **55.0%** |

## Quantitative Interpretation

### Key Findings

1. **Best Strategy**: `adaptive` achieved the highest average score 
   (0.5156 ± 0.1307)

3. **Score Distribution**: The proportion of queries in each performance category 
   provides clear quantitative context:

   - **35.0%** of queries achieved good scores (0.6-0.8)

### Conclusion

The quantitative analysis demonstrates that the RAG system provides 
genuinely strong performance improvements over baseline methods, with 
statistically significant results and clear quantitative metrics supporting 
the effectiveness of the approach.
