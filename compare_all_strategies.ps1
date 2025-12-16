# PowerShell script to compare strategies across chunk configs, k values, and baselines using LLM judge
# This script does NOT re-index data - it uses existing Pinecone indexes
#
# What this script does:
# 1. Tests 3 chunk configurations (small_chunks, medium_chunks, small_overlap)
# 2. Tests 3 k values (3, 5, 10)
# 3. Tests 3 strategies (baseline, sentence, adaptive) SEPARATELY with LLM judge
# 4. Tests 3 baselines (tfidf, keyword, retrieval_only) with retrieval metrics
# 5. Saves all results in organized folders - ONE FILE PER COMBINATION

# Configuration
$testsetFile = "tests/embedding_testset.json"
$baseOutputDir = "evaluation/comprehensive_comparison_results"
$apiKeysPath = "utils/api_keys.json"

# Define configurations
$chunkConfigs = @(
    @{name="small_chunks"; index="haifa-municipality-rag-small-chunks"},
    @{name="medium_chunks"; index="haifa-municipality-rag-medium-chunks"},
    @{name="small_overlap"; index="haifa-municipality-rag-small-overlap"}
)

$kValues = @(3, 5, 10)
$strategies = @("baseline", "sentence", "adaptive")

# Create base output directories
New-Item -ItemType Directory -Force -Path $baseOutputDir | Out-Null
$llmJudgeDir = Join-Path $baseOutputDir "llm_judge"
$baselinesDir = Join-Path $baseOutputDir "baselines"
New-Item -ItemType Directory -Force -Path $llmJudgeDir | Out-Null
New-Item -ItemType Directory -Force -Path $baselinesDir | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Comprehensive Strategy Comparison" -ForegroundColor Cyan
Write-Host "Using LLM Judge Evaluation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Running each combination separately:" -ForegroundColor Yellow
Write-Host "  - 3 chunk configs × 3 k values × 3 strategies = 27 LLM judge evaluations" -ForegroundColor Yellow
Write-Host "  - 3 chunk configs × 3 k values = 9 baseline evaluations" -ForegroundColor Yellow
Write-Host ""

$totalRuns = $chunkConfigs.Count * $kValues.Count * $strategies.Count
$currentRun = 0
$skippedCount = 0
$completedCount = 0

# Loop through ALL combinations - each one separately
foreach ($chunkConfig in $chunkConfigs) {
    $chunkName = $chunkConfig.name
    $indexName = $chunkConfig.index
    
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Chunk Config: $chunkName" -ForegroundColor Yellow
    Write-Host "Index: $indexName" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($k in $kValues) {
        Write-Host "  Testing k=$k..." -ForegroundColor Green
        
        # Run LLM judge evaluation for EACH strategy separately
        foreach ($strategy in $strategies) {
            $currentRun++
            
            # Create output directory for THIS specific combination in llm_judge subfolder
            $outputDir = Join-Path $llmJudgeDir "$chunkName`_k$k`_$strategy"
            New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
            
            # Check if this combination is already completed
            $resultsFile = Join-Path $outputDir "llm_judge_statistics.csv"
            
            if (Test-Path $resultsFile) {
                $skippedCount++
                Write-Host "    [$currentRun/$totalRuns] $chunkName / k=$k / strategy=$strategy [SKIPPED - Already completed]" -ForegroundColor Yellow
                continue
            }
            
            Write-Host "    [$currentRun/$totalRuns] $chunkName / k=$k / strategy=$strategy" -ForegroundColor Cyan
            
            # Run LLM judge for THIS ONE strategy only
            python evaluation/run_llm_judge_evaluation.py `
                --testset_file $testsetFile `
                --output_dir $outputDir `
                --strategies $strategy `
                --top_k $k `
                --index_name $indexName `
                --api_keys_path $apiKeysPath
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "      [ERROR] Failed for $chunkName k=$k strategy=$strategy" -ForegroundColor Red
            } else {
                $completedCount++
                Write-Host "      [OK] Completed $chunkName k=$k strategy=$strategy" -ForegroundColor Green
            }
        }
        
        Write-Host ""
    }
}

Write-Host ""
Write-Host "LLM Judge Summary: $completedCount completed, $skippedCount skipped" -ForegroundColor Cyan
Write-Host ""

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Baseline Evaluations" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Now run baseline comparisons for each chunk config and k value
$baselineSkippedCount = 0
$baselineCompletedCount = 0

foreach ($chunkConfig in $chunkConfigs) {
    $chunkName = $chunkConfig.name
    $indexName = $chunkConfig.index
    
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Baseline Evaluations: $chunkName" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($k in $kValues) {
        Write-Host "  Testing k=$k with baselines..." -ForegroundColor Green
        
        # Create output directory for baseline results in baselines subfolder
        $baselineOutputDir = Join-Path $baselinesDir "$chunkName`_k$k`_baselines"
        New-Item -ItemType Directory -Force -Path $baselineOutputDir | Out-Null
        
        # Check if baseline evaluation is already completed
        $baselineResultsFile = Join-Path $baselineOutputDir "strategy_statistics.csv"
        
        if (Test-Path $baselineResultsFile) {
            $baselineSkippedCount++
            Write-Host "    [SKIPPED - Already completed]" -ForegroundColor Yellow
            Write-Host ""
            continue
        }
        
        # Run evaluation with baselines included (all strategies + 3 baselines)
        Write-Host "    Running evaluation with baselines..." -ForegroundColor Cyan
        
        $strategiesArg = $strategies -join " "
        python evaluation/generate_evaluation_results.py `
            --strategies $strategiesArg `
            --top_k $k `
            --include_baselines `
            --testset_file $testsetFile `
            --output_dir $baselineOutputDir `
            --index_name $indexName `
            --api_keys_path $apiKeysPath
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    [ERROR] Failed to run baseline evaluation for $chunkName k=$k" -ForegroundColor Red
            Write-Host "    Continuing with next configuration..." -ForegroundColor Yellow
        } else {
            $baselineCompletedCount++
            Write-Host "    [OK] Completed baseline evaluation for $chunkName k=$k" -ForegroundColor Green
        }
        
        Write-Host ""
    }
}

Write-Host ""
Write-Host "Baseline Summary: $baselineCompletedCount completed, $baselineSkippedCount skipped" -ForegroundColor Cyan
Write-Host ""

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Generating Summary Report" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create a summary script to aggregate results
$baseOutputDirNormalized = $baseOutputDir -replace '\\', '/'
$summaryScript = @"
import pandas as pd
import json
from pathlib import Path
import glob
import os

base_dir = Path(r'$baseOutputDirNormalized')
results_summary = []

print('Collecting results...')

# Collect LLM judge results - now each combination has its own folder
llm_judge_dir = base_dir / 'llm_judge'
for chunk_config in ['small_chunks', 'medium_chunks', 'small_overlap']:
    for k in [3, 5, 10]:
        for strategy in ['baseline', 'sentence', 'adaptive']:
            stats_file = llm_judge_dir / f'{chunk_config}_k{k}_{strategy}' / 'llm_judge_statistics.csv'
            if stats_file.exists():
                try:
                    df = pd.read_csv(stats_file)
                    for _, row in df.iterrows():
                        strategy_name = row['strategy'] if 'strategy' in row else row.get('config', strategy)
                        results_summary.append({
                            'chunk_config': chunk_config,
                            'k': k,
                            'evaluation_type': 'llm_judge',
                            'strategy': strategy_name,
                            'correctness_mean': row.get('correctness_mean', 0),
                            'faithfulness_mean': row.get('faithfulness_mean', 0),
                            'completeness_mean': row.get('completeness_mean', 0),
                            'conciseness_mean': row.get('conciseness_mean', 0),
                            'overall_mean': row.get('overall_mean', 0),
                        })
                except Exception as e:
                    print(f'Error reading {stats_file}: {e}')

# Collect baseline results
baselines_dir = base_dir / 'baselines'
for chunk_config in ['small_chunks', 'medium_chunks', 'small_overlap']:
    for k in [3, 5, 10]:
        stats_file = baselines_dir / f'{chunk_config}_k{k}_baselines' / 'strategy_statistics.csv'
        if stats_file.exists():
            try:
                df = pd.read_csv(stats_file)
                for _, row in df.iterrows():
                    results_summary.append({
                        'chunk_config': chunk_config,
                        'k': k,
                        'evaluation_type': 'retrieval_metrics',
                        'strategy': row['strategy'],
                        'avg_score_mean': row.get('avg_score_mean', 0),
                        'namespace_correct_mean': row.get('namespace_correct_mean', 0),
                        'precision_mean': row.get('precision_mean', None),
                        'recall_mean': row.get('recall_mean', None),
                    })
            except Exception as e:
                print(f'Error reading {stats_file}: {e}')

# Save summary
if len(results_summary) > 0:
    summary_df = pd.DataFrame(results_summary)
    summary_file = base_dir / 'comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f'\nSummary saved to: {summary_file}')
    print(f'Total results collected: {len(results_summary)}')
    
    # Print top performers
    print('\n=== Top 10 Performers by Overall Score (LLM Judge) ===')
    llm_results = summary_df[summary_df['evaluation_type'] == 'llm_judge']
    if len(llm_results) > 0:
        top_overall = llm_results.nlargest(10, 'overall_mean')
        print(top_overall[['chunk_config', 'k', 'strategy', 'overall_mean', 'correctness_mean', 'faithfulness_mean']].to_string(index=False))
    else:
        print('No LLM judge results found.')
    
    # Print baseline comparison
    print('\n=== Baseline Comparison (Retrieval Metrics) ===')
    baseline_results = summary_df[summary_df['evaluation_type'] == 'retrieval_metrics']
    if len(baseline_results) > 0:
        baseline_summary = baseline_results.groupby('strategy').agg({
            'avg_score_mean': 'mean',
            'namespace_correct_mean': 'mean',
            'precision_mean': 'mean',
            'recall_mean': 'mean'
        }).round(4)
        print(baseline_summary.to_string())
    else:
        print('No baseline results found.')
else:
    print('No results found to summarize.')
"@

$summaryScript | Out-File -FilePath "$baseOutputDir/generate_summary.py" -Encoding UTF8

Write-Host "Running summary generation..." -ForegroundColor Cyan
python "$baseOutputDir/generate_summary.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Summary generated successfully" -ForegroundColor Green
} else {
    Write-Host "[WARN] Summary generation had issues, but individual results are available" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Evaluation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: $baseOutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "Directory structure:" -ForegroundColor Cyan
Write-Host "  $baseOutputDir/" -ForegroundColor Yellow
Write-Host "    ├── llm_judge/" -ForegroundColor Cyan
Write-Host "    │   ├── {chunk_config}_k{k}_{strategy}/     - LLM judge results for EACH combination" -ForegroundColor Cyan
Write-Host "    │   └── (27 folders total)" -ForegroundColor Cyan
Write-Host "    ├── baselines/" -ForegroundColor Cyan
Write-Host "    │   ├── {chunk_config}_k{k}_baselines/       - Baseline comparison results" -ForegroundColor Cyan
Write-Host "    │   └── (9 folders total)" -ForegroundColor Cyan
Write-Host "    └── comparison_summary.csv                  - Aggregated summary of all results" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total combinations:" -ForegroundColor Cyan
Write-Host "  - LLM Judge: $completedCount completed, $skippedCount skipped (out of 27 total)" -ForegroundColor Cyan
Write-Host "  - Baselines: $baselineCompletedCount completed, $baselineSkippedCount skipped (out of 9 total)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Skipped combinations already have results files. Delete them to re-run." -ForegroundColor Gray
Write-Host ""
