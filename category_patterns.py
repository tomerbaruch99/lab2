from typing import List

"""
Category Synonyms and Patterns for Targeted Retrieval

Provides synonyms and retrieval patterns for each CUAD category to improve
presence checking accuracy.
"""

# Category synonyms and patterns for targeted retrieval
CATEGORY_PATTERNS = {
    "Document Name": {
        "synonyms": ["agreement name", "contract name", "document title", "agreement title"],
        "patterns": [r"this\s+(?:agreement|contract|document)", r"entitled\s+['\"](.+?)['\"]", r"known\s+as\s+['\"](.+?)['\"]"]
    },
    "Parties": {
        "synonyms": ["parties to the agreement", "contracting parties", "the parties", "party names"],
        "patterns": [r"between\s+(.+?)\s+and\s+(.+?)(?:\s+\(|,|$)", r"party\s+(?:a|b|one|two)", r"hereinafter\s+(?:referred\s+to\s+as\s+)?['\"](.+?)['\"]"]
    },
    "Agreement Date": {
        "synonyms": ["execution date", "signed date", "date of agreement", "contract date"],
        "patterns": [r"dated\s+(?:as\s+of\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})", r"executed\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", r"agreement\s+date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"]
    },
    "Effective Date": {
        "synonyms": ["commencement date", "start date", "effective as of", "effective from"],
        "patterns": [r"effective\s+(?:date|as\s+of|from)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", r"commence(?:ment|s)?\s+(?:date|on)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"]
    },
    "Expiration Date": {
        "synonyms": ["end date", "termination date", "expiry date", "expires on"],
        "patterns": [r"expir(?:es|ation|y)\s+(?:date|on)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", r"terminat(?:es|ion)?\s+(?:date|on)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", r"perpetual|evergreen|indefinite"]
    },
    "Renewal Term": {
        "synonyms": ["renewal period", "renewal duration", "automatic renewal", "renewal term"],
        "patterns": [r"renew(?:al|s)?\s+(?:for|period|term)[:\s]+(\d+)\s+(?:days?|months?|years?)", r"automatic\s+renewal", r"renewal\s+term[:\s]+(\d+)\s+(?:days?|months?|years?)"]
    },
    "Notice Period To Terminate Renewal": {
        "synonyms": ["notice to terminate renewal", "renewal termination notice", "notice period for renewal"],
        "patterns": [r"notice\s+(?:period|of)\s+(?:to\s+)?terminat(?:e|ion)\s+renewal[:\s]+(\d+)\s+(?:days?|months?)", r"(\d+)\s+(?:days?|months?)\s+notice\s+to\s+terminate\s+renewal"]
    },
    "Governing Law": {
        "synonyms": ["jurisdiction", "applicable law", "law governing", "legal jurisdiction"],
        "patterns": [r"govern(?:ed|ing)?\s+by\s+(?:the\s+)?laws?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", r"jurisdiction[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", r"laws?\s+of\s+(?:the\s+)?(?:State\s+of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"]
    },
    "Most Favored Nation": {
        "synonyms": ["MFN", "most favored nation", "most-favored-nation", "favored nation clause"],
        "patterns": [r"most\s+favored\s+nation", r"MFN\s+clause", r"most-favored-nation"]
    },
    "Non-Compete": {
        "synonyms": ["non-compete", "noncompete", "competition restriction", "non-competition"],
        "patterns": [r"non[-\s]?compete", r"non[-\s]?competition", r"restrict(?:ion|s)?\s+on\s+compet(?:ition|ing)"]
    },
    "Exclusivity": {
        "synonyms": ["exclusive", "exclusivity", "exclusive arrangement", "exclusive right"],
        "patterns": [r"exclusiv(?:e|ity)", r"exclusive\s+(?:right|arrangement|relationship)"]
    },
    "No-Solicit Of Customers": {
        "synonyms": ["customer non-solicit", "no customer solicitation", "customer solicitation restriction"],
        "patterns": [r"no[-\s]?solicit(?:ation)?\s+of\s+customers?", r"restrict(?:ion|s)?\s+on\s+solicit(?:ing|ation)?\s+customers?", r"non[-\s]?solicit\s+customers?"]
    },
    "Competitive Restriction Exception": {
        "synonyms": ["competition exception", "restriction exception", "competitive exception"],
        "patterns": [r"exception(?:s)?\s+to\s+(?:competitive\s+)?restrictions?", r"competitive\s+restriction\s+exception"]
    },
    "No-Solicit Of Employees": {
        "synonyms": ["employee non-solicit", "no employee solicitation", "employee solicitation restriction"],
        "patterns": [r"no[-\s]?solicit(?:ation)?\s+of\s+employees?", r"restrict(?:ion|s)?\s+on\s+solicit(?:ing|ation)?\s+employees?", r"non[-\s]?solicit\s+employees?"]
    },
    "Non-Disparagement": {
        "synonyms": ["non-disparagement", "no disparagement", "disparagement restriction"],
        "patterns": [r"non[-\s]?disparagement", r"no\s+disparagement", r"disparagement\s+restriction"]
    },
    "Termination For Convenience": {
        "synonyms": ["terminate without cause", "termination without cause", "terminate at will", "convenience termination"],
        "patterns": [r"terminat(?:e|ion)\s+(?:for\s+)?convenience", r"terminat(?:e|ion)\s+without\s+cause", r"terminat(?:e|ion)\s+at\s+will"]
    },
    "Rofr/Rofo/Rofn": {
        "synonyms": ["right of first refusal", "right of first offer", "right of first negotiation", "ROFR", "ROFO", "ROFN"],
        "patterns": [r"right\s+of\s+first\s+(?:refusal|offer|negotiation)", r"ROFR|ROFO|ROFN"]
    },
    "Change Of Control": {
        "synonyms": ["change in control", "control change", "ownership change", "control event"],
        "patterns": [r"change\s+(?:of|in)\s+control", r"control\s+change", r"ownership\s+change"]
    },
    "Anti-Assignment": {
        "synonyms": ["assignment restriction", "no assignment", "assignment prohibition", "assignment consent"],
        "patterns": [r"anti[-\s]?assignment", r"assignment\s+restriction", r"no\s+assignment", r"assignment\s+(?:prohibited|requires\s+consent)"]
    },
    "Revenue/Profit Sharing": {
        "synonyms": ["revenue sharing", "profit sharing", "revenue split", "profit split"],
        "patterns": [r"revenue\s+sharing", r"profit\s+sharing", r"revenue\s+split", r"profit\s+split"]
    },
    "Price Restrictions": {
        "synonyms": ["pricing restrictions", "price controls", "pricing limitations"],
        "patterns": [r"price\s+restrictions?", r"pricing\s+restrictions?", r"price\s+controls?"]
    },
    "Minimum Commitment": {
        "synonyms": ["minimum purchase", "minimum commitment", "minimum order", "purchase commitment"],
        "patterns": [r"minimum\s+(?:purchase|commitment|order)", r"purchase\s+commitment"]
    },
    "Volume Restriction": {
        "synonyms": ["volume limits", "volume thresholds", "volume restrictions"],
        "patterns": [r"volume\s+(?:restrictions?|limits?|thresholds?)"]
    },
    "Ip Ownership Assignment": {
        "synonyms": ["IP assignment", "intellectual property assignment", "IP ownership transfer"],
        "patterns": [r"IP\s+assignment", r"intellectual\s+property\s+assignment", r"IP\s+ownership\s+transfer"]
    },
    "Joint Ip Ownership": {
        "synonyms": ["joint IP ownership", "joint intellectual property", "co-owned IP"],
        "patterns": [r"joint\s+IP\s+ownership", r"joint\s+intellectual\s+property", r"co[-\s]?owned\s+IP"]
    },
    "License Grant": {
        "synonyms": ["license granted", "grant of license", "licensing", "license provision"],
        "patterns": [r"license\s+grant(?:ed)?", r"grant\s+(?:of\s+)?license", r"licensing\s+provision"]
    },
    "Non-Transferable License": {
        "synonyms": ["non-transferable license", "non-transferable", "license not transferable"],
        "patterns": [r"non[-\s]?transferable\s+license", r"license\s+(?:is\s+)?not\s+transferable"]
    },
    "Affiliate License-Licensor": {
        "synonyms": ["affiliate license licensor", "licensor affiliate license", "affiliate licensing"],
        "patterns": [r"affiliate\s+license[-\s]?licensor", r"licensor\s+affiliate\s+license"]
    },
    "Affiliate License-Licensee": {
        "synonyms": ["affiliate license licensee", "licensee affiliate license"],
        "patterns": [r"affiliate\s+license[-\s]?licensee", r"licensee\s+affiliate\s+license"]
    },
    "Unlimited/All-You-Can-Eat-License": {
        "synonyms": ["unlimited license", "all you can eat license", "enterprise license", "unrestricted license"],
        "patterns": [r"unlimited\s+license", r"all[-\s]?you[-\s]?can[-\s]?eat\s+license", r"enterprise\s+license"]
    },
    "Irrevocable Or Perpetual License": {
        "synonyms": ["irrevocable license", "perpetual license", "permanent license", "evergreen license"],
        "patterns": [r"irrevocable\s+license", r"perpetual\s+license", r"permanent\s+license", r"evergreen\s+license"]
    },
    "Source Code Escrow": {
        "synonyms": ["source code escrow", "code escrow", "escrow agreement"],
        "patterns": [r"source\s+code\s+escrow", r"code\s+escrow", r"escrow\s+agreement"]
    },
    "Post-Termination Services": {
        "synonyms": ["post-termination services", "services after termination", "transition services"],
        "patterns": [r"post[-\s]?termination\s+services?", r"services?\s+after\s+termination", r"transition\s+services?"]
    },
    "Audit Rights": {
        "synonyms": ["audit rights", "right to audit", "audit provision", "audit access"],
        "patterns": [r"audit\s+rights?", r"right\s+to\s+audit", r"audit\s+provision"]
    },
    "Uncapped Liability": {
        "synonyms": ["uncapped liability", "unlimited liability", "no liability cap"],
        "patterns": [r"uncapped\s+liability", r"unlimited\s+liability", r"no\s+liability\s+cap"]
    },
    "Cap On Liability": {
        "synonyms": ["liability cap", "liability limit", "maximum liability", "liability ceiling"],
        "patterns": [r"liability\s+cap", r"liability\s+limit", r"maximum\s+liability", r"liability\s+ceiling"]
    },
    "Liquidated Damages": {
        "synonyms": ["liquidated damages", "termination fees", "penalty fees", "early termination fee"],
        "patterns": [r"liquidated\s+damages", r"termination\s+fees?", r"penalty\s+fees?", r"early\s+termination\s+fee"]
    },
    "Warranty Duration": {
        "synonyms": ["warranty period", "warranty term", "warranty duration", "warranty length"],
        "patterns": [r"warranty\s+(?:period|term|duration|length)[:\s]+(\d+)\s+(?:days?|months?|years?)", r"(\d+)\s+(?:days?|months?|years?)\s+warranty"]
    },
    "Insurance": {
        "synonyms": ["insurance requirement", "insurance coverage", "insurance provision", "insurance obligation"],
        "patterns": [r"insurance\s+(?:requirement|coverage|provision|obligation)", r"required\s+insurance"]
    },
    "Covenant Not To Sue": {
        "synonyms": ["covenant not to sue", "no-sue covenant", "sue restriction"],
        "patterns": [r"covenant\s+not\s+to\s+sue", r"no[-\s]?sue\s+covenant", r"sue\s+restriction"]
    },
    "Third Party Beneficiary": {
        "synonyms": ["third party beneficiary", "third-party beneficiary", "beneficiary rights"],
        "patterns": [r"third[-\s]?party\s+beneficiary", r"beneficiary\s+rights?"]
    }
}


def get_category_queries(category: str) -> List[str]:
    """
    Get retrieval queries for a category using synonyms and patterns.
    
    Args:
        category: CUAD category name
    
    Returns:
        List of query strings for targeted retrieval
    """
    queries = []
    
    if category in CATEGORY_PATTERNS:
        patterns = CATEGORY_PATTERNS[category]
        
        # Add synonyms as queries
        queries.extend(patterns.get("synonyms", []))
        
        # Add category name itself
        queries.append(category)
        
        # Add pattern-based queries (simplified - would extract from patterns)
        queries.append(f"what is the {category.lower()}")
        queries.append(f"does this agreement have {category.lower()}")
    else:
        # Fallback: use category name
        queries.append(category)
        queries.append(f"what is the {category.lower()}")
    
    return queries

