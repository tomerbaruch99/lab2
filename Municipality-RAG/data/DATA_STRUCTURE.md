# Data Structure Guide

This document describes the expected structure of JSON data files used for indexing.

## Article Data Structure (`article_meta_data.json`)

```json
[
  {
    "page_link": "https://www.haifa.muni.il/article/123",
    "categories": ["services", "municipal-services"],
    "title": "כותרת המאמר",
    "subtitle": "תת כותרת (אופציונלי)",
    "article_text": "תוכן המאמר המלא...",
    "image_links": ["https://example.com/image.jpg"],
    "links": ["https://example.com/link"]
  }
]
```

## Service Data Structure (`service_meta_data.json`)

```json
[
  {
    "page_link": "https://www.haifa.muni.il/services/license",
    "categories": ["municipal-services", "licenses"],
    "title": "רישיון עסק",
    "description": "תיאור השירות...",
    "details": [
      "פרט 1",
      "פרט 2"
    ]
  }
]
```

## Announcement Data Structure (`announcement_meta_data.json`)

```json
[
  {
    "page_link": "https://www.haifa.muni.il/announcements/456",
    "categories": ["announcements", "news"],
    "title": "כותרת ההודעה",
    "date": "01/01/2024",
    "content": "תוכן ההודעה המלא..."
  }
]
```

## Namespace Assignment

Documents are automatically assigned to namespaces based on:
1. URL contains city name (haifa/tel-aviv)
2. Title contains city name
3. Categories match namespace keywords

Supported namespaces:
- `haifa` - Haifa-specific information
- `tel-aviv` - Tel Aviv-specific information
- `municipal-services` - General municipal services
- `city-planning` - Urban planning and construction
- `waste-management` - Waste and recycling
- `transportation` - Transportation and parking
- `education` - Educational services
- `culture-recreation` - Culture and recreation
- `social-services` - Social services
- `general` - General/default namespace

