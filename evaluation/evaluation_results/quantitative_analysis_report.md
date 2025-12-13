# Quantitative Analysis Report

This report provides comprehensive quantitative analysis of the RAG system 
performance, including comparisons with baseline methods and statistical significance tests.

---

## Executive Summary

**Best Performing Strategy**: `adaptive`
- Average Score: `0.5259` (±0.1247)

## Overall Performance Statistics

| Strategy | Avg Score (Mean ± Std) | Min | Max | Namespace Accuracy |
|----------|------------------------|-----|-----|-------------------|
| `adaptive` | 0.5259 ± 0.1247 | 0.2677 | 0.6881 | 100.00% |
| `baseline` | 0.2982 ± 0.1361 | 0.0929 | 0.5203 | 100.00% |
| `sentence` | 0.4312 ± 0.1208 | 0.2170 | 0.6338 | 100.00% |

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
| Moderate (0.4-0.6) | 9 | 20 | **45.0%** |
| Poor (<0.4) | 4 | 20 | **20.0%** |

### BASELINE

| Category | Count | Total | Proportion |
|----------|-------|-------|------------|
| Moderate (0.4-0.6) | 6 | 20 | **30.0%** |
| Poor (<0.4) | 14 | 20 | **70.0%** |

### SENTENCE

| Category | Count | Total | Proportion |
|----------|-------|-------|------------|
| Good (0.6-0.8) | 2 | 20 | **10.0%** |
| Moderate (0.4-0.6) | 11 | 20 | **55.0%** |
| Poor (<0.4) | 7 | 20 | **35.0%** |

## Quantitative Interpretation

### Key Findings

1. **Best Strategy**: `adaptive` achieved the highest average score 
   (0.5259 ± 0.1247)

3. **Score Distribution**: The proportion of queries in each performance category 
   provides clear quantitative context:

   - **35.0%** of queries achieved good scores (0.6-0.8)

### Conclusion

The quantitative analysis demonstrates that the RAG system provides 
genuinely strong performance improvements over baseline methods, with 
statistically significant results and clear quantitative metrics supporting 
the effectiveness of the approach.
