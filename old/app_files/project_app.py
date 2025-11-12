"""
Streamlit UI for Contract Clause Q&A Assistant and Contract Reviewer

Product 1: Q&A Assistant - Ask questions about contract clauses
Product 2: Contract Reviewer - Review contracts for missing/unusual items
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import os

from qa_pipeline import QAPipeline, answer_question
from contract_reviewer import ContractReviewer

# Page configuration
st.set_page_config(
    page_title="Contract Clause Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

# Check for required environment variables
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ **Missing API Key**: Please set `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable to use this app.")
    st.stop()

# Initialize session state
if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None
if "contract_reviewer" not in st.session_state:
    st.session_state.contract_reviewer = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "contract_text" not in st.session_state:
    st.session_state.contract_text = None


def initialize_qa_pipeline():
    """Initialize Q&A pipeline (lazy loading)."""
    if st.session_state.qa_pipeline is None:
        with st.spinner("Loading Q&A pipeline..."):
            try:
                st.session_state.qa_pipeline = QAPipeline()
            except Exception as e:
                st.error(f"Error loading Q&A pipeline: {e}")
                return None
    return st.session_state.qa_pipeline


def initialize_contract_reviewer():
    """Initialize contract reviewer (lazy loading)."""
    if st.session_state.contract_reviewer is None:
        with st.spinner("Loading contract reviewer..."):
            try:
                st.session_state.contract_reviewer = ContractReviewer()
            except Exception as e:
                st.error(f"Error loading contract reviewer: {e}")
                return None
    return st.session_state.contract_reviewer


def display_qa_answer(result: Dict[str, Any]):
    """Display Q&A answer with citations and evidence cards."""
    # Short answer (bold)
    st.markdown("### 📝 Short Answer")
    st.markdown(f"**{result['short_answer']}**")
    
    # Rationale
    st.markdown("### 💭 Rationale")
    st.markdown(result['rationale'])
    
    # Citations with evidence cards
    if result.get('citations'):
        st.markdown("### 📚 Evidence Cards")
        
        for i, citation in enumerate(result['citations'], 1):
            with st.expander(f"Clause {i}: {citation.get('citation', 'Unknown')}"):
                # Highlight span if available
                text = citation.get('text', '')
                start_idx = citation.get('start_idx')
                end_idx = citation.get('end_idx')
                
                if start_idx is not None and end_idx is not None:
                    # Highlight the span
                    before = text[:start_idx]
                    highlight = text[start_idx:end_idx]
                    after = text[end_idx:]
                    st.markdown(f"{before}<mark>{highlight}</mark>{after}", unsafe_allow_html=True)
                else:
                    st.markdown(text)
                
                # Redaction badge
                if citation.get('has_redaction'):
                    st.warning("⚠️ This clause contains redactions")
                
                # Metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"**File:** {citation.get('filename', 'Unknown')}")
                with col2:
                    st.caption(f"**Category:** {citation.get('category', 'Unknown')}")
        
        # Conflicting clauses
        if result.get('conflicting_clauses'):
            st.markdown("### ⚠️ Conflicting Clauses")
            st.warning("Multiple clauses found with conflicting information:")
            for conf_clause in result['conflicting_clauses']:
                with st.expander(f"Conflicting: {conf_clause.get('citation', 'Unknown')}"):
                    st.markdown(conf_clause.get('text', ''))
                    st.caption(f"Category: {conf_clause.get('category', 'Unknown')}")
    
    # Suggested categories if answer not found
    if result.get('suggested_categories'):
        st.markdown("### 💡 Suggested Categories")
        st.info("Answer not found. You might want to check these related categories:")
        for cat, _ in result['suggested_categories']:
            st.write(f"- {cat}")


def display_review_report(report: Dict[str, Any]):
    """Display contract review report with heatmap and findings."""
    # Review report
    st.markdown("### 📊 Review Report")
    st.markdown(report.get('review_report', 'No report generated'))
    
    # Presence map heatmap
    st.markdown("### 🗺️ Category Presence Heatmap")
    presence_map = report.get('presence_map', {})
    
    # Create heatmap data
    categories = list(presence_map.keys())
    statuses = []
    colors = []
    
    for category in categories:
        info = presence_map[category]
        if info['present']:
            statuses.append("✅ Present")
            colors.append("green")
        else:
            statuses.append("❌ Missing")
            colors.append("red")
    
    # Display as dataframe with color coding
    heatmap_df = pd.DataFrame({
        "Category": categories,
        "Status": statuses,
        "Confidence": [presence_map[cat]['confidence'] for cat in categories]
    })
    
    st.dataframe(
        heatmap_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Outliers
    outliers = report.get('outliers', {})
    if outliers:
        st.markdown("### 📈 Statistical Outliers")
        for category, outlier_info in outliers.items():
            with st.expander(f"⚠️ {category}: {outlier_info.get('explanation', '')}"):
                st.write(f"**Value:** {outlier_info.get('value', 'N/A')}")
                st.write(f"**95th Percentile:** {outlier_info.get('percentile_95', 'N/A')}")
                if outlier_info.get('is_extreme'):
                    st.error("This is an extreme outlier (>99th percentile)")
    
    # Consistency issues
    consistency_issues = report.get('consistency_issues', [])
    if consistency_issues:
        st.markdown("### 🔍 Consistency Issues")
        for issue in consistency_issues:
            severity_color = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(issue.get('severity', 'low'), "⚪")
            
            st.markdown(f"{severity_color} **{issue.get('category', 'Unknown')}**: {issue.get('issue', '')}")
            
            if issue.get('suggested_clauses'):
                st.caption(f"Suggested related clauses: {', '.join(issue['suggested_clauses'])}")
    
    # Extracted values
    extracted_values = report.get('extracted_values', {})
    if extracted_values:
        st.markdown("### 📋 Extracted Structured Values")
        values_df = pd.DataFrame({
            "Category": list(extracted_values.keys()),
            "Value": list(extracted_values.values())
        })
        st.dataframe(values_df, use_container_width=True, hide_index=True)


# Main app
st.title("📄 Contract Clause Q&A Assistant")
st.markdown("Ask questions about contract clauses or review contracts for missing/unusual items")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Contract")
    uploaded_file = st.file_uploader(
        "Upload a contract (PDF/TXT)",
        type=['txt', 'pdf'],
        help="Upload a contract file to ask questions about it or review it"
    )
    
    if uploaded_file is not None:
        # Read file content
        if uploaded_file.type == "text/plain":
            contract_text = str(uploaded_file.read(), "utf-8")
            st.session_state.contract_text = contract_text
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"✅ Loaded: {uploaded_file.name}")
        else:
            st.warning("PDF support coming soon. Please upload a TXT file.")
    
    st.markdown("---")
    st.markdown("### Settings")
    use_extractive = st.checkbox("Use Extractive Reader", value=True, help="Use extractive reader for exact answer spans")
    k_clauses = st.slider("Number of clauses to retrieve", 1, 10, 5)

# Main tabs
tab1, tab2 = st.tabs(["🔍 Q&A Assistant", "📊 Contract Reviewer"])

# Tab 1: Q&A Assistant
with tab1:
    st.header("Ask Questions About Contract Clauses")
    
    # Initialize pipeline
    pipeline = initialize_qa_pipeline()
    
    if pipeline is None:
        st.error("Could not initialize Q&A pipeline. Please check your configuration.")
    else:
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., 'What is the governing law?' or 'Can this agreement be terminated without cause?'",
            help="Ask any question about contract clauses"
        )
        
        if st.button("🔍 Answer Question", type="primary"):
            if question:
                with st.spinner("Retrieving clauses and generating answer..."):
                    try:
                        result = pipeline.answer(
                            question=question,
                            filename=st.session_state.uploaded_filename,
                            k=k_clauses,
                            use_extractive=use_extractive
                        )
                        
                        display_qa_answer(result)
                        
                    except Exception as e:
                        st.error(f"Error answering question: {e}")
            else:
                st.warning("Please enter a question")

# Tab 2: Contract Reviewer
with tab2:
    st.header("Review Contract for Missing/Unusual Items")
    
    # Initialize reviewer
    reviewer = initialize_contract_reviewer()
    
    if reviewer is None:
        st.error("Could not initialize contract reviewer. Please check your configuration.")
    else:
        if st.session_state.contract_text is None:
            st.info("👈 Please upload a contract file in the sidebar first")
        else:
            contract_type = st.selectbox(
                "Contract Type (for conditional priors):",
                options=["unknown", "service agreement", "license agreement", "joint venture", 
                        "consulting agreement", "agency agreement", "collaboration agreement"],
                help="Select contract type for better outlier detection"
            )
            
            if st.button("🔍 Review Contract", type="primary"):
                with st.spinner("Reviewing contract... This may take a few minutes."):
                    try:
                        report = reviewer.review(
                            contract_text=st.session_state.contract_text,
                            filename=st.session_state.uploaded_filename or "uploaded_contract.txt",
                            contract_type=contract_type
                        )
                        
                        display_review_report(report)
                        
                    except Exception as e:
                        st.error(f"Error reviewing contract: {e}")

# Footer
st.markdown("---")
st.markdown("**Contract Clause Q&A Assistant** | Built with RAG, Extractive QA, and LLM")

