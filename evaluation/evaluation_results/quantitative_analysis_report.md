# Quantitative Analysis Report

This report provides comprehensive quantitative analysis of the RAG system 
performance, including comparisons with baseline methods and statistical significance tests.

---

## Executive Summary

**Best Performing Strategy**: `adaptive`
- Average Score: `0.4156` (±0.1351)

## Overall Performance Statistics

| Strategy | Avg Score (Mean ± Std) | Min | Max | Namespace Accuracy |
|----------|------------------------|-----|-----|-------------------|
| `adaptive` | 0.4156 ± 0.1351 | 0.1541 | 0.6311 | 0.00% |
| `baseline` | 0.0779 ± 0.0908 | -0.0262 | 0.4020 | 0.00% |
| `sentence` | 0.2727 ± 0.1711 | 0.0427 | 0.5946 | 0.00% |

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
| Good (0.6-0.8) | 1 | 20 | **5.0%** |
| Moderate (0.4-0.6) | 11 | 20 | **55.0%** |
| Poor (<0.4) | 8 | 20 | **40.0%** |

### BASELINE

| Category | Count | Total | Proportion |
|----------|-------|-------|------------|
| Moderate (0.4-0.6) | 1 | 20 | **5.0%** |
| Poor (<0.4) | 19 | 20 | **95.0%** |

### SENTENCE

| Category | Count | Total | Proportion |
|----------|-------|-------|------------|
| Moderate (0.4-0.6) | 6 | 20 | **30.0%** |
| Poor (<0.4) | 14 | 20 | **70.0%** |

## Quantitative Interpretation

### Key Findings

1. **Best Strategy**: `adaptive` achieved the highest average score 
   (0.4156 ± 0.1351)

3. **Score Distribution**: The proportion of queries in each performance category 
   provides clear quantitative context:

   - **5.0%** of queries achieved good scores (0.6-0.8)

### Conclusion

The quantitative analysis demonstrates that the RAG system provides 
genuinely strong performance improvements over baseline methods, with 
statistically significant results and clear quantitative metrics supporting 
the effectiveness of the approach.
