"""
FastAPI Service for Sentiment Analysis
Exposes REST API endpoints for the Python sentiment engine
ADVANCED VERSION with Transformer Models, Batch Processing, and Trend Analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import uvicorn

from sentiment_analyzer import SentimentAnalyzer
from dataset_generator import DatasetGenerator
from batch_processor import BatchProcessor
from trend_analyzer import TrendAnalyzer
from persistent_pool import get_persistent_pool  # Import persistent pool

# Optional: Import transformer analyzer (requires large dependencies)
try:
    from transformer_analyzer import TransformerSentimentAnalyzer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("⚠️ Transformer models not available. Install transformers and torch for premium features.")

app = FastAPI(
    title="Advanced Sentiment Analysis Service",
    version="2.0.0",
    description="Premium sentiment analysis with transformers, batch processing, and trend analysis"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
vader_analyzer = SentimentAnalyzer()
generator = DatasetGenerator()
batch_processor = BatchProcessor(vader_analyzer)
trend_analyzer = TrendAnalyzer()

# Initialize transformer analyzer if available
transformer_analyzer = None
if TRANSFORMER_AVAILABLE:
    try:
        print("🚀 Loading transformer model (this may take a moment)...")
        transformer_analyzer = TransformerSentimentAnalyzer()
        batch_processor_transformer = BatchProcessor(transformer_analyzer)
        print("✅ Transformer model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load transformer model: {e}")
        TRANSFORMER_AVAILABLE = False

# Persistent pool will be initialized on startup (Windows requires this)
persistent_pool = None


@app.on_event("startup")
async def startup_event():
    """Initialize persistent worker pool on server startup"""
    global persistent_pool
    print("🔥 Initializing persistent worker pool...")
    persistent_pool = get_persistent_pool()
    print("✅ Persistent pool ready for zero-overhead parallel processing!")


class AnalyzeRequest(BaseModel):
    """Request model for sentiment analysis"""
    texts: List[str]
    parallel: Optional[bool] = True
    num_workers: Optional[int] = None
    model: Optional[str] = "vader"  # "vader" or "transformer"


class GenerateDatasetRequest(BaseModel):
    """Request model for dataset generation"""
    count: int = 10000
    distribution: Optional[dict] = None


class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis"""
    texts: List[str]
    timestamps: Optional[List[str]] = None
    interval: Optional[str] = "hour"
    model: Optional[str] = "vader"


class AnalyzeResponse(BaseModel):
    """Response model for sentiment analysis"""
    positive: int
    negative: int
    neutral: int
    processing_time: float
    method: str
    total_processed: int
    model_used: Optional[str] = "vader"
    avg_confidence: Optional[float] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Advanced Sentiment Analysis Engine",
        "status": "running",
        "version": "2.0.0",
        "features": {
            "vader_model": True,
            "transformer_model": TRANSFORMER_AVAILABLE,
            "batch_processing": True,
            "trend_analysis": True,
            "file_upload": True
        }
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Analyze sentiment with model selection
    
    Args:
        request: AnalyzeRequest with texts, processing options, and model choice
    
    Returns:
        Sentiment analysis results
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        # Select analyzer
        if request.model == "transformer":
            if not TRANSFORMER_AVAILABLE:
                raise HTTPException(
                    status_code=400,
                    detail="Transformer model not available. Install transformers and torch."
                )
            current_analyzer = transformer_analyzer
        else:
            current_analyzer = vader_analyzer
        
        # Analyze
        if request.parallel:
            result = current_analyzer.analyze_parallel(
                request.texts,
                num_workers=request.num_workers
            )
        else:
            result = current_analyzer.analyze_sequential(request.texts)
        
        return AnalyzeResponse(
            positive=result['summary']['positive'],
            negative=result['summary']['negative'],
            neutral=result['summary']['neutral'],
            processing_time=round(result['processing_time'], 3),
            method=result['method'],
            total_processed=result['total_processed'],
            model_used=request.model,
            avg_confidence=result.get('avg_confidence')
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/compare")
async def compare_performance(request: AnalyzeRequest):
    """
    Compare sequential vs parallel performance
    
    Returns:
        Both sequential and parallel results with speedup metrics
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        result = vader_analyzer.compare_performance(request.texts, num_workers=request.num_workers)
        
        return {
            "sequential": {
                "positive": result['sequential']['summary']['positive'],
                "negative": result['sequential']['summary']['negative'],
                "neutral": result['sequential']['summary']['neutral'],
                "processing_time": round(result['sequential']['processing_time'], 3),
                "method": "sequential"
            },
            "parallel": {
                "positive": result['parallel']['summary']['positive'],
                "negative": result['parallel']['summary']['negative'],
                "neutral": result['parallel']['summary']['neutral'],
                "processing_time": round(result['parallel']['processing_time'], 3),
                "method": result['parallel']['method'],
                "num_workers": result['parallel'].get('num_workers', 0)
            },
            "speedup": round(result['speedup'], 2),
            "improvement_percent": round(result['improvement_percent'], 1),
            "total_processed": len(request.texts),
            "recommendation": result.get('recommendation', '')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-dataset")
async def generate_dataset(request: GenerateDatasetRequest):
    """
    Generate synthetic dataset for testing
    
    Args:
        request: GenerateDatasetRequest with count and distribution
    
    Returns:
        Generated texts
    """
    try:
        # Generate dataset using the correct method name
        df = generator.generate(
            count=request.count,
            distribution=request.distribution
        )
        
        # Convert DataFrame to list of texts
        texts = df['text'].tolist()
        
        return {
            "texts": texts,
            "count": len(texts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-file")
async def upload_file_for_analysis(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    model: str = Form("vader")
):
    """
    Upload a file (TXT, CSV, XLSX) for batch sentiment analysis
    
    Args:
        file: Uploaded file
        text_column: Column name containing text (for CSV/Excel)
        model: Model to use ("vader" or "transformer")
    
    Returns:
        Sentiment analysis results
    """
    try:
        # Read file content
        content = await file.read()
        
        # Select processor
        if model == "transformer":
            if not TRANSFORMER_AVAILABLE:
                raise HTTPException(
                    status_code=400,
                    detail="Transformer model not available"
                )
            processor = batch_processor_transformer
        else:
            processor = batch_processor
        
        # Process file
        result = processor.process_file(
            file_content=content,
            filename=file.filename,
            text_column=text_column
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trend-analysis")
async def analyze_trend(request: TrendAnalysisRequest):
    """
    Perform trend analysis on sentiment data over time
    
    Args:
        request: TrendAnalysisRequest with texts, timestamps, and options
    
    Returns:
        Comprehensive trend analysis
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        # Select analyzer
        if request.model == "transformer":
            if not TRANSFORMER_AVAILABLE:
                raise HTTPException(status_code=400, detail="Transformer model not available")
            current_analyzer = transformer_analyzer
        else:
            current_analyzer = vader_analyzer
        
        # Analyze sentiments
        result = current_analyzer.analyze_parallel(request.texts)
        sentiments = result['detailed_results']
        
        # Parse timestamps if provided
        timestamps = None
        if request.timestamps:
            try:
                timestamps = [datetime.fromisoformat(ts) for ts in request.timestamps]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO format.")
        
        # Generate trend summary
        summary = trend_analyzer.generate_summary(sentiments, timestamps)
        
        return summary
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_available_models():
    """
    Get information about available models
    
    Returns:
        Available models and their capabilities
    """
    models = [
        {
            "name": "vader",
            "display_name": "VADER (Fast)",
            "available": True,
            "type": "rule-based",
            "speed": "very_fast",
            "accuracy": "good",
            "description": "Fast rule-based sentiment analyzer, ideal for large datasets"
        }
    ]
    
    if TRANSFORMER_AVAILABLE:
        models.append({
            "name": "transformer",
            "display_name": "DistilBERT (Premium)",
            "available": True,
            "type": "deep_learning",
            "speed": "moderate",
            "accuracy": "excellent",
            "description": "Advanced transformer model with higher accuracy, best for quality analysis"
        })
    else:
        models.append({
            "name": "transformer",
            "display_name": "DistilBERT (Premium)",
            "available": False,
            "type": "deep_learning",
            "reason": "Dependencies not installed"
        })
    
    return {
        "models": models,
        "default": "vader"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Advanced Sentiment Analysis Service")
    print("=" * 60)
    print(f"✅ VADER Model: Available")
    print(f"{'✅' if TRANSFORMER_AVAILABLE else '⚠️'} Transformer Model: {'Available' if TRANSFORMER_AVAILABLE else 'Not Available'}")
    print(f"✅ Batch Processing: Available")
    print(f"✅ Trend Analysis: Available")
    print("=" * 60)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
