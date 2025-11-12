"""
Product 2: RAG-driven Contract Reviewer

Takes a complete contract and identifies missing or unusual items compared to CUAD patterns.
Includes:
- Presence checks (binary, per category)
- Norm-aware outlier detection
- Consistency checks (intra-document)
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

from rag_retriever import retrieve
from extractive_reader import ExtractiveReader
from rag_generator import RAGGenerator
from query_understanding import CUAD_CATEGORIES, QUESTION_TEMPLATES
from category_patterns import get_category_queries
from consts import GEMINI_MODEL

# Import category definitions from data_preparation
try:
    from data_preparation import CUAD_CATEGORIES as CATEGORY_DEFS
except ImportError:
    # Fallback: create minimal category definitions
    CATEGORY_DEFS = [{"name": cat, "answer_type": "text", "group": None} for cat in CUAD_CATEGORIES]

# Redaction pattern
RE_REDACT = re.compile(r"(\*{2,}|_{2,}|<omitted>|\[\])", re.IGNORECASE)


class PresenceChecker:
    """Checks presence of each CUAD category in a contract."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize presence checker.
        
        Args:
            confidence_threshold: Minimum confidence to consider a category present
        """
        self.confidence_threshold = confidence_threshold
    
    def check_presence(
        self,
        contract_text: str,
        filename: str,
        categories: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check presence of each category in the contract.
        
        Args:
            contract_text: Full contract text
            filename: Contract filename
            categories: List of categories to check (defaults to all CUAD categories)
        
        Returns:
            Dictionary mapping category -> {
                "present": bool,
                "confidence": float,
                "top_evidence": List[Dict]  # Top 2 clauses
            }
        """
        if categories is None:
            categories = CUAD_CATEGORIES
        
        presence_map = {}
        
        for category in categories:
            # Use category synonyms + patterns for targeted retrieval
            category_queries = get_category_queries(category)
            
            # Try multiple queries and take best results
            all_clauses = []
            for query in category_queries[:3]:  # Use top 3 query variations
                clauses = retrieve(
                    query=query,
                    k_dense=3,
                    k_bm25=3,
                    final_k=2,  # Top 2 evidence snippets per query
                    filename=filename,
                    category=category,
                    use_query_understanding=False  # Already have category
                )
                all_clauses.extend(clauses)
            
            # Deduplicate and take top 2
            seen_texts = set()
            unique_clauses = []
            for clause in all_clauses:
                clause_text = clause.get("text", clause.get("context", ""))
                if clause_text not in seen_texts:
                    seen_texts.add(clause_text)
                    unique_clauses.append(clause)
                    if len(unique_clauses) >= 2:
                        break
            
            clauses = unique_clauses
            
            # Determine presence based on retrieval results
            present = len(clauses) > 0
            confidence = 1.0 if present else 0.0
            
            # If present, calculate confidence based on relevance scores
            if present:
                # Simple heuristic: if we got results, it's likely present
                # In production, would use actual relevance scores
                confidence = 0.8 if len(clauses) >= 2 else 0.6
            
            top_evidence = []
            for i, clause in enumerate(clauses[:2], 1):
                top_evidence.append({
                    "clause_num": i,
                    "text": clause.get("text", clause.get("context", "")),
                    "filename": clause.get("filename", filename),
                    "category": clause.get("category", category),
                    "has_redaction": bool(RE_REDACT.search(clause.get("text", "")))
                })
            
            presence_map[category] = {
                "present": present and confidence >= self.confidence_threshold,
                "confidence": confidence,
                "top_evidence": top_evidence
            }
        
        return presence_map


class OutlierDetector:
    """Detects statistical outliers in structured contract values."""
    
    def __init__(self, cuad_data_path: str = "cuad_prepared_data/cuad_long_clauses.parquet"):
        """
        Initialize outlier detector.
        
        Args:
            cuad_data_path: Path to CUAD data for learning priors
        """
        self.cuad_data_path = cuad_data_path
        self.priors = self._load_priors()
    
    def _load_priors(self) -> Dict[str, Dict[str, Any]]:
        """Load empirical priors from CUAD data."""
        try:
            df = pd.read_parquet(self.cuad_data_path)
            
            priors = {}
            
            # Group by category and contract_type
            for category in CUAD_CATEGORIES:
                cat_data = df[df["category"] == category]
                
                if len(cat_data) == 0:
                    continue
                
                # Extract structured values based on answer type
                category_def = next(
                    (c for c in CATEGORY_DEFS if c["name"] == category),
                    None
                )
                
                if not category_def:
                    continue
                
                answer_type = category_def.get("answer_type", "text")
                
                # Calculate statistics per contract type
                priors[category] = {}
                
                for contract_type in df["contract_type"].unique():
                    type_data = cat_data[cat_data["contract_type"] == contract_type]
                    
                    if len(type_data) == 0:
                        continue
                    
                    # Extract and analyze values based on type
                    values = self._extract_values(type_data["answer"].tolist(), answer_type)
                    
                    if values:
                        priors[category][contract_type] = {
                            "mean": np.mean(values),
                            "median": np.median(values),
                            "std": np.std(values),
                            "percentiles": {
                                "25": np.percentile(values, 25),
                                "75": np.percentile(values, 75),
                                "95": np.percentile(values, 95),
                                "99": np.percentile(values, 99)
                            },
                            "min": np.min(values),
                            "max": np.max(values)
                        }
            
            return priors
        
        except Exception as e:
            print(f"[WARN] Could not load priors: {e}. Using empty priors.")
            return {}
    
    def _extract_values(self, answers: List[str], answer_type: str) -> List[float]:
        """Extract numeric values from answers based on type."""
        values = []
        
        for answer in answers:
            if not answer or pd.isna(answer):
                continue
            
            answer_str = str(answer).lower()
            
            if answer_type == "duration":
                # Extract days, months, years
                days_match = re.search(r'(\d+)\s*(?:days?|d)', answer_str)
                if days_match:
                    values.append(float(days_match.group(1)))
                else:
                    months_match = re.search(r'(\d+)\s*(?:months?|mo)', answer_str)
                    if months_match:
                        values.append(float(months_match.group(1)) * 30)  # Convert to days
                    else:
                        years_match = re.search(r'(\d+)\s*(?:years?|yr|y)', answer_str)
                        if years_match:
                            values.append(float(years_match.group(1)) * 365)  # Convert to days
            
            elif answer_type in ["date", "date_or_perpetual"]:
                # Extract year
                year_match = re.search(r'(\d{4})', answer_str)
                if year_match:
                    values.append(float(year_match.group(1)))
        
        return values
    
    def detect_outliers(
        self,
        extracted_values: Dict[str, Any],
        contract_type: str = "unknown"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers in extracted values.
        
        Args:
            extracted_values: Dictionary mapping category -> extracted value
            contract_type: Contract type for conditional priors
        
        Returns:
            Dictionary mapping category -> {
                "is_outlier": bool,
                "percentile": float,
                "explanation": str
            }
        """
        outliers = {}
        
        for category, value in extracted_values.items():
            if category not in self.priors:
                continue
            
            # Get priors for this contract type, or fallback to all types
            type_priors = self.priors[category].get(contract_type)
            if not type_priors:
                # Use average across all types
                all_priors = list(self.priors[category].values())
                if not all_priors:
                    continue
                type_priors = {
                    "mean": np.mean([p["mean"] for p in all_priors]),
                    "percentiles": {
                        "95": np.mean([p["percentiles"]["95"] for p in all_priors]),
                        "99": np.mean([p["percentiles"]["99"] for p in all_priors])
                    }
                }
            
            # Check if value is an outlier
            numeric_value = self._extract_numeric_value(value, category)
            
            if numeric_value is None:
                continue
            
            percentile_95 = type_priors["percentiles"]["95"]
            percentile_99 = type_priors["percentiles"]["99"]
            
            is_outlier = numeric_value > percentile_95
            is_extreme = numeric_value > percentile_99
            
            if is_outlier:
                explanation = f"{category} value ({value}) is above 95th percentile"
                if is_extreme:
                    explanation += " (extreme outlier)"
                explanation += f" for {contract_type} contracts in CUAD"
                
                outliers[category] = {
                    "is_outlier": True,
                    "is_extreme": is_extreme,
                    "value": value,
                    "percentile_95": percentile_95,
                    "percentile_99": percentile_99,
                    "explanation": explanation
                }
        
        return outliers
    
    def _extract_numeric_value(self, value: Any, category: str) -> Optional[float]:
        """Extract numeric value from answer."""
        if value is None:
            return None
        
        value_str = str(value).lower()
        
        # Extract days
        days_match = re.search(r'(\d+)\s*(?:days?|d)', value_str)
        if days_match:
            return float(days_match.group(1))
        
        # Extract months
        months_match = re.search(r'(\d+)\s*(?:months?|mo)', value_str)
        if months_match:
            return float(months_match.group(1)) * 30
        
        # Extract years
        years_match = re.search(r'(\d+)\s*(?:years?|yr|y)', value_str)
        if years_match:
            return float(years_match.group(1)) * 365
        
        # Extract year from date
        year_match = re.search(r'(\d{4})', value_str)
        if year_match:
            return float(year_match.group(1))
        
        return None


class ConsistencyChecker:
    """Checks intra-document consistency (dates, clause coherence)."""
    
    def __init__(self):
        """Initialize consistency checker."""
        pass
    
    def check_consistency(
        self,
        extracted_values: Dict[str, Any],
        clauses: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Check consistency of extracted values.
        
        Args:
            extracted_values: Dictionary mapping category -> extracted value
            clauses: List of all clauses
        
        Returns:
            List of consistency issues
        """
        issues = []
        
        # Date logic checks
        effective_date = extracted_values.get("Effective Date")
        expiration_date = extracted_values.get("Expiration Date")
        agreement_date = extracted_values.get("Agreement Date")
        renewal_term = extracted_values.get("Renewal Term")
        
        # Check: Effective Date <= Expiration Date
        if effective_date and expiration_date:
            eff_dt = self._parse_date(effective_date)
            exp_dt = self._parse_date(expiration_date)
            
            if eff_dt and exp_dt and eff_dt > exp_dt:
                issues.append({
                    "type": "date_logic",
                    "category": "Effective Date vs Expiration Date",
                    "issue": f"Effective Date ({effective_date}) is after Expiration Date ({expiration_date})",
                    "severity": "high"
                })
        
        # Check: Agreement Date <= Effective Date (if both present)
        # Effective mirrors Agreement Date if defined that way
        if agreement_date and effective_date:
            agr_dt = self._parse_date(agreement_date)
            eff_dt = self._parse_date(effective_date)
            
            if agr_dt and eff_dt:
                if agr_dt > eff_dt:
                    issues.append({
                        "type": "date_logic",
                        "category": "Agreement Date vs Effective Date",
                        "issue": f"Agreement Date ({agreement_date}) is after Effective Date ({effective_date})",
                        "severity": "medium"
                    })
                # Note: If they match, that's expected behavior (Effective mirrors Agreement)
        
        # Check: Renewal adds to Expiration Date
        if renewal_term and expiration_date:
            exp_dt = self._parse_date(expiration_date)
            if exp_dt:
                # Extract renewal duration (simplified - would parse properly)
                renewal_days = self._extract_duration_days(renewal_term)
                if renewal_days:
                    # Calculate new expiration after renewal
                    from datetime import timedelta
                    new_expiration = exp_dt + timedelta(days=renewal_days)
                    # This is informational - renewal extends expiration
                    # Would check if there's a new expiration date that matches
        
        # Clause coherence checks
        # If License Grant present, check for related clauses
        if extracted_values.get("License Grant") == "Yes":
            related_clauses = [
                "Non-Transferable License",
                "Affiliate License-Licensor",
                "Affiliate License-Licensee",
                "Irrevocable Or Perpetual License",
                "Unlimited/All-You-Can-Eat-License"
            ]
            
            missing_related = []
            for rel_clause in related_clauses:
                if rel_clause not in extracted_values:
                    missing_related.append(rel_clause)
            
            if missing_related:
                issues.append({
                    "type": "clause_coherence",
                    "category": "License Grant",
                    "issue": f"License Grant present but related clauses not found: {', '.join(missing_related)}",
                    "severity": "low",
                    "suggested_clauses": missing_related
                })
        
        # Change of Control vs Anti-Assignment alignment
        if extracted_values.get("Change Of Control") == "Yes":
            if extracted_values.get("Anti-Assignment") != "Yes":
                issues.append({
                    "type": "clause_coherence",
                    "category": "Change Of Control",
                    "issue": "Change Of Control present but Anti-Assignment not explicitly stated",
                    "severity": "medium"
                })
        
        return issues
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str or pd.isna(date_str):
            return None
        
        date_formats = [
            "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
            "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except:
                continue
        
        # Try to extract year
        year_match = re.search(r'(\d{4})', str(date_str))
        if year_match:
            try:
                return datetime(int(year_match.group(1)), 1, 1)
            except:
                pass
        
        return None
    
    def _extract_duration_days(self, duration_str: str) -> Optional[int]:
        """Extract duration in days from string."""
        if not duration_str or pd.isna(duration_str):
            return None
        
        duration_lower = str(duration_str).lower()
        
        # Extract days
        days_match = re.search(r'(\d+)\s*(?:days?|d)', duration_lower)
        if days_match:
            return int(days_match.group(1))
        
        # Extract months
        months_match = re.search(r'(\d+)\s*(?:months?|mo)', duration_lower)
        if months_match:
            return int(months_match.group(1)) * 30
        
        # Extract years
        years_match = re.search(r'(\d+)\s*(?:years?|yr|y)', duration_lower)
        if years_match:
            return int(years_match.group(1)) * 365
        
        return None


class ContractReviewer:
    """
    Main contract reviewer that combines presence checks, outlier detection,
    and consistency checks to produce a review report.
    """
    
    def __init__(
        self,
        extractive_reader_model_dir: str = "models/extractive_reader",
        gemini_model: str = GEMINI_MODEL
    ):
        """
        Initialize contract reviewer.
        
        Args:
            extractive_reader_model_dir: Path to extractive reader model
            gemini_model: Gemini model name for review report generation (defaults to GEMINI_MODEL from consts)
        """
        self.presence_checker = PresenceChecker()
        self.outlier_detector = OutlierDetector()
        self.consistency_checker = ConsistencyChecker()
        
        # Initialize extractive reader for value extraction
        try:
            self.extractive_reader = ExtractiveReader(model_dir=extractive_reader_model_dir)
        except Exception as e:
            print(f"[WARN] Could not load extractive reader: {e}. Continuing without it.")
            self.extractive_reader = None
        
        self.rag_generator = RAGGenerator(model=gemini_model)
    
    def review(
        self,
        contract_text: str,
        filename: str,
        contract_type: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Review a contract and produce findings.
        
        Args:
            contract_text: Full contract text
            filename: Contract filename
            contract_type: Contract type (for conditional priors)
        
        Returns:
            Review report dictionary
        """
        # Step 1: Sectionization (simplified - would use heading detection)
        # For now, we'll work with the full text
        
        # Step 2: Category-driven retrieval (presence checks)
        print("[Step 2] Checking category presence...")
        presence_map = self.presence_checker.check_presence(
            contract_text=contract_text,
            filename=filename
        )
        
        # Step 3: Extract structured values
        print("[Step 3] Extracting structured values...")
        extracted_values = self._extract_structured_values(
            contract_text=contract_text,
            filename=filename,
            presence_map=presence_map
        )
        
        # Step 4: Outlier detection
        print("[Step 4] Detecting outliers...")
        outliers = self.outlier_detector.detect_outliers(
            extracted_values=extracted_values,
            contract_type=contract_type
        )
        
        # Step 5: Consistency checks
        print("[Step 5] Checking consistency...")
        # Get all clauses for consistency checking
        all_clauses = []
        for category, info in presence_map.items():
            if info["present"]:
                all_clauses.extend(info["top_evidence"])
        
        consistency_issues = self.consistency_checker.check_consistency(
            extracted_values=extracted_values,
            clauses=all_clauses
        )
        
        # Step 6: Generate review report using LLM
        print("[Step 6] Generating review report...")
        review_report = self._generate_review_report(
            presence_map=presence_map,
            extracted_values=extracted_values,
            outliers=outliers,
            consistency_issues=consistency_issues,
            contract_type=contract_type
        )
        
        return {
            "presence_map": presence_map,
            "extracted_values": extracted_values,
            "outliers": outliers,
            "consistency_issues": consistency_issues,
            "review_report": review_report,
            "contract_type": contract_type,
            "filename": filename
        }
    
    def _extract_structured_values(
        self,
        contract_text: str,
        filename: str,
        presence_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured values for present categories."""
        extracted = {}
        
        for category, info in presence_map.items():
            if not info["present"]:
                continue
            
            # Get category definition
            category_def = next(
                (c for c in CATEGORY_DEFS if c["name"] == category),
                None
            )
            
            if not category_def:
                continue
            
            answer_type = category_def.get("answer_type", "text")
            
            # Use extractive reader if available
            if self.extractive_reader and info["top_evidence"]:
                try:
                    question = QUESTION_TEMPLATES.get(category, f"What is the {category}?")
                    clause_text = info["top_evidence"][0]["text"]
                    
                    result = self.extractive_reader.extract_answer(question, clause_text)
                    
                    if result and result.get("answer_text"):
                        extracted[category] = result["answer_text"]
                        continue
                except Exception as e:
                    print(f"[WARN] Error extracting value for {category}: {e}. Using fallback.")
            
            # Fallback: use answer from presence check if available
            if info["top_evidence"]:
                # Try to extract from evidence
                evidence_text = info["top_evidence"][0]["text"]
                # Simple extraction (would be improved with better parsing)
                extracted[category] = evidence_text[:100]  # Simplified
        
        return extracted
    
    def _generate_review_report(
        self,
        presence_map: Dict[str, Dict[str, Any]],
        extracted_values: Dict[str, Any],
        outliers: Dict[str, Dict[str, Any]],
        consistency_issues: List[Dict[str, Any]],
        contract_type: str
    ) -> str:
        """Generate review report using LLM."""
        # Build prompt for reviewer
        prompt = self._build_reviewer_prompt(
            presence_map=presence_map,
            extracted_values=extracted_values,
            outliers=outliers,
            consistency_issues=consistency_issues,
            contract_type=contract_type
        )
        
        # Generate report
        try:
            report = self.rag_generator.generate(
                question="Generate a contract review report based on the findings below.",
                clauses=[{"text": prompt, "filename": "review", "category": "Review"}],
                max_tokens=1000,
                include_format_policy=True
            )
            return report
        except Exception as e:
            print(f"[WARN] Error generating review report: {e}")
            return f"Error generating review report: {str(e)}"
    
    def _build_reviewer_prompt(
        self,
        presence_map: Dict[str, Dict[str, Any]],
        extracted_values: Dict[str, Any],
        outliers: Dict[str, Dict[str, Any]],
        consistency_issues: List[Dict[str, Any]],
        contract_type: str
    ) -> str:
        """Build reviewer prompt skeleton."""
        prompt_parts = [
            "CONTRACT REVIEW INPUTS:",
            "",
            f"Contract Type: {contract_type}",
            "",
            "PRESENCE MAP (top-2 clauses per category):"
        ]
        
        # Add presence map
        for category, info in presence_map.items():
            status = "Present (ok)" if info["present"] else "Missing"
            prompt_parts.append(f"\n{category}: {status}")
            
            if info["present"] and info["top_evidence"]:
                for i, evidence in enumerate(info["top_evidence"][:2], 1):
                    prompt_parts.append(f"  Evidence {i}: {evidence['text'][:200]}...")
        
        # Add extracted values
        prompt_parts.append("\n\nEXTRACTED STRUCTURED VALUES:")
        for category, value in extracted_values.items():
            prompt_parts.append(f"{category}: {value}")
        
        # Add outliers
        if outliers:
            prompt_parts.append("\n\nOUTLIERS:")
            for category, outlier_info in outliers.items():
                prompt_parts.append(f"{category}: {outlier_info['explanation']}")
        
        # Add consistency issues
        if consistency_issues:
            prompt_parts.append("\n\nCONSISTENCY ISSUES:")
            for issue in consistency_issues:
                prompt_parts.append(f"- {issue['issue']} (Severity: {issue['severity']})")
        
        # Add tasks
        prompt_parts.extend([
            "",
            "REVIEWER TASKS:",
            "1) Mark each category: Present (ok), Missing, or Unusual (with explanation)",
            "2) Cite evidence: filename + clause snippet IDs",
            "3) Explicitly state weak or redacted evidence",
            "4) Never infer redacted content",
            "",
            "Generate a findings table and explanation."
        ])
        
        return "\n".join(prompt_parts)

