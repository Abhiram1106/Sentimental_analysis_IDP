# 📡 API Documentation - Sentiment Analysis Platform

## Overview

This document provides comprehensive documentation for all API endpoints in the Advanced Sentiment Analysis Platform. The platform exposes two sets of APIs:

1. **Backend API** (Node.js/Express) - Port 5000
2. **Python ML Engine API** (FastAPI) - Port 8000

---

## Table of Contents

- [Authentication](#authentication)
- [Base URLs](#base-urls)
- [Backend API Endpoints](#backend-api-endpoints)
- [Python ML Engine Endpoints](#python-ml-engine-endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

---

## Authentication

**Current Version**: No authentication required (suitable for development/internal use)

**Production Recommendation**: Implement JWT or OAuth2 authentication before deploying to production.

---

## Base URLs

| Service | Base URL | Environment |
|---------|----------|-------------|
| Backend API | `http://localhost:5000/api` | Development |
| Python ML Engine | `http://localhost:8000` | Development |

---

## Backend API Endpoints

### 1. Health Check

**Endpoint**: `GET /api/health`  
**Description**: Check if the backend service is running  
**Authentication**: None

#### Response
```json
{
  "status": "running",
  "service": "Sentiment Analysis Backend",
  "timestamp": "2026-02-19T10:30:00.000Z"
}
```

---

### 2. Analyze Sentiment

**Endpoint**: `POST /api/analyze`  
**Description**: Analyze sentiment of multiple texts with optional parallel processing  
**Authentication**: None

#### Request Body
```json
{
  "texts": ["I love this product!", "This is terrible"],
  "parallel": true,
  "model": "vader"
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | array[string] | ✅ Yes | - | Array of texts to analyze |
| `parallel` | boolean | ❌ No | `true` | Enable parallel processing |
| `model` | string | ❌ No | `"vader"` | Model to use: `"vader"` or `"transformer"` |

#### Response
```json
{
  "success": true,
  "data": {
    "total_processed": 2,
    "positive": 1,
    "negative": 1,
    "neutral": 0,
    "processing_time": 0.234,
    "method": "parallel (process, optimized concurrent)",
    "model_used": "vader",
    "avg_confidence": 0.89,
    "details": [
      {
        "text": "I love this product!",
        "sentiment": "positive",
        "confidence": 0.92,
        "scores": {
          "neg": 0.0,
          "neu": 0.192,
          "pos": 0.808,
          "compound": 0.6369
        }
      },
      {
        "text": "This is terrible",
        "sentiment": "negative",
        "confidence": 0.86,
        "scores": {
          "neg": 0.778,
          "neu": 0.222,
          "pos": 0.0,
          "compound": -0.5423
        }
      }
    ]
  }
}
```

#### Status Codes
- `200 OK`: Successful analysis
- `400 Bad Request`: Invalid input (empty texts array)
- `500 Internal Server Error`: Python service unavailable or processing error

---

### 3. Compare Processing Methods

**Endpoint**: `POST /api/analyze/compare`  
**Description**: Compare sequential vs parallel processing performance  
**Authentication**: None

#### Request Body
```json
{
  "texts": ["Sample text 1", "Sample text 2", "..."],
  "model": "vader"
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | array[string] | ✅ Yes | - | Array of texts to analyze |
| `model` | string | ❌ No | `"vader"` | Model to use |

#### Response
```json
{
  "success": true,
  "comparison": {
    "sequential": {
      "time": 4.523,
      "positive": 150,
      "negative": 50,
      "neutral": 300,
      "method": "sequential"
    },
    "parallel": {
      "time": 1.024,
      "positive": 150,
      "negative": 50,
      "neutral": 300,
      "method": "parallel (process, optimized concurrent)"
    },
    "speedup": "4.42x faster",
    "time_saved": "3.499 seconds"
  }
}
```

---

### 4. Generate Dataset

**Endpoint**: `POST /api/generate-dataset`  
**Description**: Generate synthetic test dataset with sentiment labels  
**Authentication**: None

#### Request Body
```json
{
  "count": 1000,
  "distribution": {
    "positive": 0.4,
    "negative": 0.3,
    "neutral": 0.3
  }
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `count` | integer | ❌ No | `1000` | Number of texts to generate (100-50000) |
| `distribution` | object | ❌ No | Equal split | Sentiment distribution (values sum to 1.0) |

#### Response
```json
{
  "success": true,
  "dataset": {
    "total_generated": 1000,
    "distribution": {
      "positive": 400,
      "negative": 300,
      "neutral": 300
    },
    "texts": [
      "This is amazing!",
      "I hate this product.",
      "It's okay.",
      "..."
    ],
    "generation_time": 0.156
  }
}
```

---

### 5. Upload File for Batch Analysis

**Endpoint**: `POST /api/upload`  
**Description**: Upload CSV, TXT, or XLSX file for batch sentiment analysis  
**Authentication**: None  
**Content-Type**: `multipart/form-data`

#### Form Data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ Yes | File to upload (max 50MB) |
| `model` | string | ❌ No | Model to use: `"vader"` or `"transformer"` |
| `textColumn` | string | ❌ No | Column name for structured files (CSV/XLSX) |

#### Supported File Formats
- **TXT**: One text per line
- **CSV**: Comma-separated values with headers
- **XLSX**: Excel spreadsheet

#### Response
```json
{
  "success": true,
  "results": {
    "total_processed": 5000,
    "positive": 2100,
    "negative": 1200,
    "neutral": 1700,
    "processing_time": 2.345,
    "model_used": "vader",
    "avg_confidence": 0.87,
    "details": [
      {
        "text": "Sample text from file",
        "sentiment": "positive",
        "confidence": 0.91,
        "scores": { "neg": 0.0, "neu": 0.15, "pos": 0.85, "compound": 0.7351 }
      }
    ]
  }
}
```

#### Status Codes
- `200 OK`: File processed successfully
- `400 Bad Request`: No file uploaded or invalid format
- `413 Payload Too Large`: File exceeds 50MB
- `500 Internal Server Error`: Processing failed

---

### 6. Trend Analysis

**Endpoint**: `POST /api/trend-analysis`  
**Description**: Analyze sentiment trends over time with predictions  
**Authentication**: None

#### Request Body
```json
{
  "texts": [
    "Great product!",
    "Getting better",
    "Amazing experience!"
  ],
  "timestamps": [
    "2026-02-01T10:00:00Z",
    "2026-02-02T10:00:00Z",
    "2026-02-03T10:00:00Z"
  ],
  "interval": "day",
  "model": "vader"
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | array[string] | ✅ Yes | - | Array of texts to analyze |
| `timestamps` | array[string] | ❌ No | Auto-generated | ISO 8601 timestamps for each text |
| `interval` | string | ❌ No | `"hour"` | Time grouping: `"minute"`, `"hour"`, `"day"`, `"week"` |
| `model` | string | ❌ No | `"vader"` | Model to use |

#### Response
```json
{
  "success": true,
  "trend_analysis": {
    "total_analyzed": 3,
    "time_range": {
      "start": "2026-02-01T10:00:00Z",
      "end": "2026-02-03T10:00:00Z"
    },
    "overall_sentiment": "positive",
    "trend_direction": "improving",
    "correlation": {
      "value": 0.89,
      "strength": "strong"
    },
    "volatility": {
      "score": 0.12,
      "interpretation": "stable"
    },
    "data_points": [
      {
        "timestamp": "2026-02-01T10:00:00Z",
        "avg_sentiment": 0.65,
        "sentiment_label": "positive",
        "count": 1
      }
    ],
    "predictions": {
      "next_3_periods": [0.72, 0.75, 0.78],
      "trend": "increasing",
      "confidence": 0.84
    },
    "moving_average": [0.65, 0.68, 0.71],
    "peak_period": {
      "timestamp": "2026-02-03T10:00:00Z",
      "sentiment": 0.79
    }
  }
}
```

---

### 7. Get Analysis Results

**Endpoint**: `GET /api/results`  
**Description**: Retrieve analysis history from MongoDB  
**Authentication**: None

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | ❌ No | `10` | Number of results to return |

#### Response
```json
{
  "success": true,
  "count": 10,
  "results": [
    {
      "_id": "65abc123...",
      "totalProcessed": 100,
      "results": {
        "positive": 45,
        "negative": 30,
        "neutral": 25
      },
      "processingTime": 1.234,
      "method": "parallel (process, optimized concurrent)",
      "metadata": {
        "model": "vader",
        "avgConfidence": 0.88
      },
      "timestamp": "2026-02-19T10:00:00.000Z"
    }
  ]
}
```

#### Status Codes
- `200 OK`: Results retrieved successfully
- `500 Internal Server Error`: Database error (MongoDB not connected)

---

### 8. Get Statistics

**Endpoint**: `GET /api/stats`  
**Description**: Get aggregate statistics from all analyses  
**Authentication**: None

#### Response
```json
{
  "success": true,
  "statistics": {
    "totalAnalyses": 150,
    "totalTextsProcessed": 45000,
    "avgProcessingTime": 2.345,
    "mostUsedMethod": "parallel (process, optimized concurrent)",
    "sentimentDistribution": {
      "positive": 18000,
      "negative": 12000,
      "neutral": 15000
    }
  }
}
```

---

### 9. Get Available Models

**Endpoint**: `GET /api/models`  
**Description**: List available ML models and their status  
**Authentication**: None

#### Response
```json
{
  "success": true,
  "models": {
    "vader": {
      "available": true,
      "name": "VADER Sentiment Analyzer",
      "version": "3.3.2",
      "type": "lexicon-based",
      "speed": "very fast",
      "accuracy": "good",
      "best_for": "social media, large datasets"
    },
    "transformer": {
      "available": true,
      "name": "DistilBERT",
      "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
      "type": "transformer",
      "speed": "moderate",
      "accuracy": "excellent",
      "best_for": "detailed analysis, accuracy-critical tasks"
    }
  }
}
```

---

## Python ML Engine Endpoints

### Base URL: `http://localhost:8000`

### 1. Service Information

**Endpoint**: `GET /`  
**Description**: Get service information and available endpoints

#### Response
```json
{
  "service": "Advanced Sentiment Analysis Service",
  "version": "2.0.0",
  "models_available": {
    "vader": true,
    "transformer": true
  },
  "endpoints": [
    "/analyze",
    "/analyze/compare",
    "/generate-dataset",
    "/upload-file",
    "/trend-analysis",
    "/models"
  ],
  "documentation": "/docs"
}
```

---

### 2. Analyze Sentiment (ML Engine)

**Endpoint**: `POST /analyze`  
**Description**: Direct access to ML engine for sentiment analysis

#### Request Body
```json
{
  "texts": ["Sample text"],
  "parallel": true,
  "num_workers": null,
  "model": "vader"
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | array[string] | ✅ Yes | - | Texts to analyze |
| `parallel` | boolean | ❌ No | `true` | Enable parallel processing |
| `num_workers` | integer | ❌ No | CPU count | Number of worker processes |
| `model` | string | ❌ No | `"vader"` | `"vader"` or `"transformer"` |

#### Response
```json
{
  "total_processed": 1,
  "positive": 1,
  "negative": 0,
  "neutral": 0,
  "processing_time": 0.012,
  "method": "parallel (process, optimized concurrent)",
  "model_used": "vader",
  "avg_confidence": 0.92,
  "details": [
    {
      "text": "Sample text",
      "sentiment": "positive",
      "confidence": 0.92,
      "scores": {
        "neg": 0.0,
        "neu": 0.1,
        "pos": 0.9,
        "compound": 0.7506
      }
    }
  ]
}
```

---

### 3. Compare Processing Methods (ML Engine)

**Endpoint**: `POST /analyze/compare`  
**Description**: Compare sequential vs parallel performance

#### Request Body
```json
{
  "texts": ["Text 1", "Text 2"],
  "model": "vader"
}
```

#### Response
*(Same structure as Backend API /api/analyze/compare)*

---

### 4. Generate Dataset (ML Engine)

**Endpoint**: `POST /generate-dataset`  
**Description**: Generate synthetic sentiment dataset

#### Request Body
```json
{
  "count": 5000,
  "distribution": {
    "positive": 0.4,
    "negative": 0.3,
    "neutral": 0.3
  }
}
```

#### Response
```json
{
  "total_generated": 5000,
  "distribution": {
    "positive": 2000,
    "negative": 1500,
    "neutral": 1500
  },
  "texts": ["..."],
  "generation_time": 0.345
}
```

---

### 5. Upload File (ML Engine)

**Endpoint**: `POST /upload-file`  
**Description**: Upload file for batch processing  
**Content-Type**: `multipart/form-data`

#### Form Data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ Yes | TXT, CSV, or XLSX file |
| `model` | string | ❌ No | Model to use |
| `text_column` | string | ❌ No | Column name for CSV/XLSX |

#### Response
*(Same structure as Backend API /api/upload)*

---

### 6. Trend Analysis (ML Engine)

**Endpoint**: `POST /trend-analysis`  
**Description**: Perform time-series sentiment analysis

#### Request/Response
*(Same structure as Backend API /api/trend-analysis)*

---

### 7. Get Models (ML Engine)

**Endpoint**: `GET /models`  
**Description**: Get information about available models

#### Response
```json
{
  "available_models": {
    "vader": {
      "available": true,
      "name": "VADER Sentiment Analyzer",
      "version": "3.3.2",
      "type": "lexicon-based",
      "speed": "very fast (~100K texts/sec)",
      "accuracy": "good (75-80%)",
      "memory": "~100MB",
      "best_for": ["social media", "real-time", "large datasets"]
    },
    "transformer": {
      "available": true,
      "name": "DistilBERT",
      "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
      "type": "transformer (deep learning)",
      "speed": "moderate (~5K texts/sec)",
      "accuracy": "excellent (90-95%)",
      "memory": "~1GB",
      "best_for": ["quality analysis", "complex text", "accuracy-critical"]
    }
  },
  "recommendation": "Use VADER for speed, Transformer for accuracy"
}
```

---

### 8. Interactive API Documentation

**Endpoint**: `GET /docs`  
**Description**: Access Swagger UI for interactive API testing  
**URL**: http://localhost:8000/docs

**Features**:
- Interactive request/response testing
- Schema documentation
- Example requests
- Try-it-now functionality

---

## Request/Response Examples

### Example 1: Simple Analysis (cURL)

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["I love this!", "This is bad"],
    "model": "vader"
  }'
```

### Example 2: File Upload (cURL)

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@reviews.csv" \
  -F "model=transformer" \
  -F "textColumn=review_text"
```

### Example 3: Trend Analysis (JavaScript/Axios)

```javascript
const axios = require('axios');

const response = await axios.post('http://localhost:5000/api/trend-analysis', {
  texts: ["Great!", "Better!", "Best!"],
  timestamps: [
    "2026-02-01T10:00:00Z",
    "2026-02-02T10:00:00Z",
    "2026-02-03T10:00:00Z"
  ],
  model: "vader"
});

console.log(response.data.trend_analysis.trend_direction);
// Output: "improving"
```

### Example 4: Performance Comparison (Python)

```python
import requests

response = requests.post('http://localhost:5000/api/analyze/compare', json={
    'texts': ['Sample text'] * 1000,
    'model': 'vader'
})

comparison = response.json()['comparison']
print(f"Speedup: {comparison['speedup']}")
# Output: "4.5x faster"
```

---

## Error Handling

### Error Response Format

All errors follow this structure:

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "timestamp": "2026-02-19T10:00:00.000Z"
}
```

### Common Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| `400` | Bad Request | Invalid input parameters |
| `404` | Not Found | Endpoint does not exist |
| `413` | Payload Too Large | File/request exceeds size limit |
| `422` | Unprocessable Entity | Valid syntax but invalid semantics |
| `500` | Internal Server Error | Server processing error |
| `503` | Service Unavailable | Python ML engine offline |
| `504` | Gateway Timeout | Request processing timeout (>2 min) |

### Example Error Responses

**Invalid Input**:
```json
{
  "error": "Invalid request",
  "message": "Please provide an array of texts to analyze",
  "timestamp": "2026-02-19T10:30:00.000Z"
}
```

**Service Unavailable**:
```json
{
  "error": "Python service unavailable",
  "message": "Could not connect to http://localhost:8000. Ensure the Python service is running.",
  "timestamp": "2026-02-19T10:30:00.000Z"
}
```

**Model Not Available**:
```json
{
  "error": "Model not available",
  "message": "Transformer model is not loaded. Install required dependencies.",
  "timestamp": "2026-02-19T10:30:00.000Z"
}
```

---

## Rate Limits

**Current Version**: No rate limiting implemented

**Production Recommendations**:
- Implement rate limiting: 100 requests/minute per IP
- File upload: 10 files/hour per user
- Large batch processing: 5 requests/hour per user

**Implementation Example** (Express.js):
```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 100 // 100 requests per minute
});

app.use('/api/', limiter);
```

---

## Performance Guidelines

### Best Practices

1. **Batch Processing**
   - Use `/api/upload` for >100 texts
   - Prefer CSV/XLSX over multiple API calls

2. **Model Selection**
   - VADER: <10K texts, real-time needs
   - Transformer: <5K texts, accuracy priority

3. **Parallel Processing**
   - Always enable for >100 texts
   - ~4-5x speedup on typical hardware

4. **Timeouts**
   - Set client timeout >2 minutes for large datasets
   - Default backend timeout: 120 seconds

### Performance Benchmarks

| Dataset Size | VADER (Parallel) | Transformer (Parallel) |
|--------------|------------------|------------------------|
| 100 texts    | ~0.2s            | ~2s                    |
| 1,000 texts  | ~1.2s            | ~18s                   |
| 10,000 texts | ~10s             | ~180s (3 min)          |
| 50,000 texts | ~52s             | Not recommended        |

---

## API Versioning

**Current Version**: v2.0  
**Version Strategy**: URL versioning (future)

**Future Implementation**:
- `/api/v2/analyze`
- `/api/v3/analyze`

**Backward Compatibility**: v2.0 endpoints will be maintained for 12 months after v3.0 release

---

## Additional Resources

- **Swagger UI**: http://localhost:8000/docs (Interactive API testing)
- **ReDoc**: http://localhost:8000/redoc (Alternative API documentation)
- **GitHub Repository**: [Sentimental_analysis_IDP](https://github.com/Abhiram1106/Sentimental_analysis_IDP)
- **Project Summary**: See `PROJECT_SUMMARY.md`
- **Setup Guide**: See `README.md`

---

## Support & Feedback

For API-related questions or issues:
- Check error messages for debugging hints
- Review FastAPI docs at http://localhost:8000/docs
- Open GitHub issue for bugs or feature requests
- Ensure all services are running (Frontend, Backend, Python Engine)

---

**API Documentation Version**: 2.0  
**Last Updated**: February 2026  
**Status**: ✅ Production Ready
