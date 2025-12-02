import requests
import json
import re
import os
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Function to call your API
def get_city_info(city_name: str) -> dict:
    """Call the FastAPI server to get real-time city information."""
    try:
        url = "http://localhost:8081/get_city_info"
        payload = {"city_name": city_name}
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
# 1) Validator + real-time API call
def validated_get_city_info(city_name: str) -> dict:
    """
    Fetch live municipal information for an Israeli city (Haifa or Tel Aviv).
    Only Hebrew letters and spaces are allowed in city_name.
    Returns JSON dict or an error message if validation fails.
    """
    # Hebrew Unicode block + spaces
    if not re.fullmatch(r'^[\u0590-\u05FF ]+$', city_name):
        return {
            "error": "City name must be provided in Hebrew letters only."
        }
    return get_city_info(city_name)

real_time_tool = Tool(
    name="get_city_info",
    func=validated_get_city_info,
    description=(
        "Fetch the latest municipal information for an Israeli city (Haifa or Tel Aviv). "
        "Input: city_name must be Hebrew letters only (e.g., 'חיפה' or 'תל אביב'). "
        "Returns JSON with either 'info' or 'error'."
    ),
)

# 3) Instantiate your Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    max_tokens=2048,
    max_retries=2,
    # ADC or service‐account creds via GOOGLE_APPLICATION_CREDENTIALS
)

# 4) Build the agent (English-only React agent)
agent = initialize_agent(
    tools=[real_time_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 5) Query function
def ask_bot(user_input: str) -> str:
    """
    Pass user_input (in Hebrew) to the agent.
    The agent will call get_city_info when needed.
    """
    return agent.invoke(user_input)


# Example usage (commented out):
# ask_bot("מה השירותים העירוניים בחיפה?")

