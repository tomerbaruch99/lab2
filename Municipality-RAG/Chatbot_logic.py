import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from API_tools import validated_get_city_info
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import re

# Load environment 
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding Model
device = "cuda" if SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').device == 'cuda' else "cpu"
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
class CustomEmbedding:
    def __init__(self, model):
        self.model = model
    def embed_query(self, text: str):
        return self.model.encode("query: " + text, normalize_embeddings=True, convert_to_numpy=True).tolist()
    
# Pinecone Vector Store 
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    return pc.Index(PINECONE_INDEX)

index = get_pinecone_index()
embedding = CustomEmbedding(embed_model)

# LLM (Gemini 2.5 Flash) 
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0, # it is crucial that answers will be precise and consistent. 
        max_output_tokens=1024,
        google_api_key=GOOGLE_API_KEY,
    
    )
llm = get_llm()

SYSTEM_PROMPT = """
                    You are "Municipality-RAG AI Assistant" (עוזר וירטואלי AI - מידע עירוני).

                    ## ROLE & VOICE  
                    • Provide **official, accurate guidance** drawn only from municipality websites (Haifa, Tel Aviv) and prior conversation context.  
                    • **Always answer in clear, simple Hebrew**.  
                    • Be concise, helpful, and informative; avoid jargon.

                    ## KNOWLEDGE BOUNDARIES  
                    • You may **quote, summarise, or reorder** retrieved passages, but **never invent facts** or contradict the source.  
                    • If context is insufficient or uncertain, say:  
                    > "אינני בטוח במידע הזמין. אנא נסח את השאלה מחדש או בדוק באתר העירייה הרשמי."

                    ## INTERACTION RULES  
                    • Treat every user message as a question about municipal services, information, or procedures.  
                    • If the request is ambiguous, ask a follow-up question in Hebrew.  
                    • For up-to-date requests (e.g., current information for a city), trigger fresh retrieval before answering.  
                    • Focus on **practical information** about municipal services, procedures, and resources.  
                    • Never reveal chain-of-thought, embeddings, or system details.

                   ## ANSWER FORMAT   
                    1. **Answer** - brief paragraphs or an ordered list.  
                    2. **למידע נוסף** - if links are provided to the relevant page/article, display each as Markdown in the form **"[כאן>>](https://…)"** (anchor text "כאן>>" only). Do **not** show raw URLs.

                    ## SAFETY & TONE  
                    • Maintain a professional, helpful tone appropriate for municipal services.  
                    • Exclude political commentary, rumours, or personal opinions.  
                    • Present yourself as the official assistance interface; do not mention you are an AI model.

                    ## FAIL-SAFE  
                    • If asked for information beyond municipal services, respond:  
                    > "אני יכול לעזור רק עם מידע עירוני. לשאלות אחרות, אנא פנה לגורם המתאים."
"""

#  Memory (Conversation Window) 
import streamlit as st
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=7
    )

# Hebrew Detection 
import re
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
def is_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text))

# Core Retrieval Logic 
def similarity_search(query, namespace, k):
    """Search Pinecone for relevant documents by namespace."""
    query_emb = embedding.embed_query(query)
    res = index.query(vector=query_emb, top_k=k, include_metadata=True, namespace=namespace)
    results = []
    for match in res.get('matches', []):
        page_content = match['metadata'].get('to_embed') \
            or match['metadata'].get('title', '') + "\n" + match['metadata'].get('subtitle', '')
        results.append({"page_content": page_content, "metadata": match['metadata'], "score": match['score']})
    return results


NAMESPACE_DESCRIPTIONS = {
    "haifa": "Information specific to Haifa municipality: services, procedures, departments, announcements, and city-specific information.",
    "tel-aviv": "Information specific to Tel Aviv municipality: services, procedures, departments, announcements, and city-specific information.",
    "municipal-services": "General municipal services available in both cities: permits, licenses, payments, registrations, and administrative procedures.",
    "city-planning": "Urban planning, construction permits, zoning, building regulations, and development projects.",
    "waste-management": "Garbage collection, recycling, waste disposal, environmental services, and cleanliness.",
    "transportation": "Public transportation, parking, traffic, road maintenance, and mobility services.",
    "education": "Schools, educational programs, after-school activities, and educational services provided by municipalities.",
    "culture-recreation": "Cultural events, libraries, community centers, parks, sports facilities, and recreational activities.",
    "social-services": "Social welfare, assistance programs, support services, and community resources.",
    "general": "For questions that don't clearly match any category or when classification is uncertain. Acts as a fallback."
}

# system-prompt for the namespace-classifier 
NS_PROMPT = f"""
You are *Namespace-Classifier* for the Municipality-RAG system.

TASK  
• Read the user query (in Hebrew).  
• Decide which **single** namespace it belongs to, choosing from this list:  
  {list(NAMESPACE_DESCRIPTIONS.keys())}

DEFINITIONS:  
{chr(10).join([f'- "{k}": {v}' for k,v in NAMESPACE_DESCRIPTIONS.items()])}

RULES  
1. **Output only the namespace label**, nothing else – no punctuation, no explanation, no greeting.  
2. Be highly confident in your choice.  
   • If confidence < 90 % return the label **general**.  
3. Never invent new labels, never return multiple labels.  
4. If the query is empty or not Hebrew, return **general**.  
5. If the query mentions "חיפה" or "Haifa", prefer **haifa** namespace.
6. If the query mentions "תל אביב" or "Tel Aviv", prefer **tel-aviv** namespace.
"""

real_time_tool = Tool(
    name="get_city_info",
    func=validated_get_city_info,
    description=(
        "Fetch the latest municipal information for an Israeli city (Haifa or Tel Aviv). "
        "Input: city_name must be Hebrew letters only (e.g., 'חיפה' or 'תל אביב'). "
        "Returns JSON with either 'info' or 'error'."
    )
)

# Initializing smaller model for namespace detection
ns_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=24,        
    google_api_key=GOOGLE_API_KEY
)

def classify_namespace(user_query: str) -> str:
    """Return ONLY the matching namespace string."""
    response = ns_model.invoke(
        NS_PROMPT + "\nUSER QUERY:\n" + user_query.strip()
    )
    # Safety-net: strip spaces/newlines just in case
    return response.content.strip()

# Create agent with real-time tool (just like in API_tools.py)
agent = initialize_agent(
    tools=[real_time_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def get_rewritten_query(original_query: str, memory_history: str = "") -> str:
    # --- gemini ---
    model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            max_tokens=128,
            google_api_key=GOOGLE_API_KEY
        )
    system_prompt = (""" 
        You are a Query Rewriter for a Retrieval-Augmented Generation (RAG) system.

        ## TASK
        Given:
        1. The conversation history between user and assistant (may be empty or irrelevant).
        2. The user's current query.

        You must return **one rewritten query** that is:
        - Fully self-contained (can be understood without any prior context).
        - More informative and specific than the original query if possible.
        - Suitable for semantic search in a database of municipal information documents.

        ## RULES
        - Use your judgment: if the history is relevant, incorporate it to clarify the query.  
        - If the history is irrelevant, ignore it.  
        - Return **only the rewritten query text**, without explanations, prefixes, or extra commentary.  
        - Always rewrite in English if the input is English; if the input is Hebrew, keep it in Hebrew.  

        ## EXAMPLES

        Conversation:
        User: "What services does the municipality provide?"  
        Assistant: "The municipality provides various services including permits, licenses..."  
        User: "And how do I apply?"  
        Rewritten query: "How to apply for municipal services and permits?"

        ---

        Conversation:
        User: "What are the parking regulations?"  
        Assistant: "Parking regulations vary by area..."  
        User: "And in the city center?"  
        Rewritten query: "What are the parking regulations in the city center?"

        ---

        Conversation:
        User: "Where can I pay municipal taxes?"  
        Rewritten query: "Where and how to pay municipal taxes and fees?"

    """

    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"היסטוריה:\n{memory_history}\n\nשאלה:\n{original_query}")
    ]

    rewritten = model.invoke(messages).content.strip()
    return rewritten

#  Main Chatbot Response Logic 
def get_rag_response_stream(query, memory, k, system_prompt=SYSTEM_PROMPT):
    """Yields answer in chunks for Streamlit chat streaming UI."""
    import time
    t_start = time.time()

    if not is_hebrew(query):
        polite_msg = "אשמח לעזור – נא לכתוב בבקשה את השאלה בעברית."
        memory.save_context({"question": query}, {"answer": polite_msg})
        yield polite_msg
        return

    # Try agent first - it will automatically decide if tool is needed
    try:
        agent_response = agent.invoke(query)
        
        # Extract the clean response from agent output
        if isinstance(agent_response, dict) and 'output' in agent_response:
            response_text = agent_response['output']
        else:
            response_text = str(agent_response)
        
        # Clean up the response - remove agent chain logs and get only the final answer
        if "Final Answer:" in response_text:
            response_text = response_text.split("Final Answer:")[-1].strip()
        
        # If we got a substantial Hebrew response, use it
        if response_text and len(response_text) > 20 and is_hebrew(response_text):
            memory.save_context({"question": query}, {"answer": response_text})
            # Stream word by word to match existing UI behavior
            words = response_text.split()
            for word in words:
                yield word + " "
                time.sleep(0.02)
            return
            
    except Exception as e:
        # If agent fails, continue to RAG - no error message to user
        print(f"[ERROR] Agent failed: {e}")
        pass

    history = memory.chat_memory.messages[-4:] if memory.chat_memory.messages else []
    # Convert message objects to formatted string for query rewriting
    history_str = ""
    if history:
        history_parts = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                history_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"Assistant: {msg.content}")
        history_str = "\n".join(history_parts)
    rewritten_query = get_rewritten_query(query, history_str)
    namespace = classify_namespace(rewritten_query)
    docs = similarity_search(rewritten_query, namespace=namespace, k=k)
    

    # context = "\n\n".join(doc["page_content"] for doc in docs).strip()
    context = "\n\n".join(doc["metadata"].get("content", "") for doc in docs).strip()

    # Debug
    with st.expander("📄 מסמכים שאוחזרו", expanded=False):
        st.markdown(f"**query after rephrasing:**\n\n{rewritten_query}")
        st.markdown("---")
        for i, doc in enumerate(docs, 1):
            meta = doc["metadata"]
            score = doc.get("score", 0)
            doc_id = doc.get("id", "unknown")
            # Choose best available label
            if "title" in meta and meta["title"]:
                doc_label = meta["title"]
            elif "question" in meta and meta["question"]:
                doc_label = meta["question"]
            else:
                doc_label = meta.get("page_link", "לא ידוע")
            st.markdown(f"**Similarity score:** `{score:.3f}`")
            st.markdown(f"**doc id:** `{doc_id}`")
            st.markdown(f"**doc {i}:** `{doc_label}`")
            st.markdown(f"namespace: `{namespace}`")
            single_content = meta.get("content", "")
            st.markdown(f"content: {single_content}")
            st.markdown("---")

    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content="להלן המידע שנמצא באתר העירייה הרשמי:\n" + context),
        *history,
        HumanMessage(content=query)
    ]
    full_reply = ""
    for chunk in llm.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            full_reply += chunk.content
            yield chunk.content
    memory.save_context({"question": query}, {"answer": full_reply})
    t_end = time.time()
    print(f"[DEBUG] RAG streaming response time: {t_end-t_start:.2f}s")

from UI import render_ui

# Suggestions to show as buttons in the UI
suggestions = [
    "מה השירותים העירוניים הזמינים?",
    "איך משלמים ארנונה?",
    "מה שעות פעילות העירייה?",
    "איך מקבלים רישיון עסק?",
]

# Logo image path 
LOGO_PATH = "static/logo.png"  


render_ui(
    suggestions=suggestions,
    logo_path=LOGO_PATH,
    get_response_streaming=get_rag_response_stream,
    memory=st.session_state.memory
)

