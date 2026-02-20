const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
require('dotenv').config();

const Analysis = require('./models/Analysis');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
});

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/sentiment_analysis';

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('✅ MongoDB connected successfully'))
.catch((err) => {
  console.log('⚠️  MongoDB connection failed. Running without database.');
  console.log('   Error:', err.message);
});

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'running',
    service: 'Sentiment Analysis Backend',
    timestamp: new Date().toISOString(),
  });
});

// Analyze sentiment (parallel processing)
app.post('/api/analyze', async (req, res) => {
  try {
    const { texts, parallel = true, model = 'vader', num_workers } = req.body;

    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'Please provide an array of texts to analyze',
      });
    }

    console.log(`📊 Processing ${texts.length} texts using ${parallel ? 'parallel' : 'sequential'} method with ${model} model...`);

    // Build request payload
    const payload = {
      texts,
      parallel,
      model,
    };
    if (num_workers !== undefined && num_workers !== null) {
      payload.num_workers = num_workers;
    }

    // Increase timeout for ensemble models (3 models running)
    const timeout = model === 'ensemble' ? 300000 : 120000; // 5 min for ensemble, 2 min for others

    // Call Python service
    const response = await axios.post(`${PYTHON_SERVICE_URL}/analyze`, payload, {
      timeout,
    });

    const result = response.data;

    // Save to MongoDB (if connected)
    try {
      const analysis = new Analysis({
        totalProcessed: result.total_processed,
        results: {
          positive: result.positive,
          negative: result.negative,
          neutral: result.neutral,
        },
        processingTime: result.processing_time,
        method: result.method,
        metadata: {
          model: result.model_used,
          avgConfidence: result.avg_confidence,
        },
      });

      await analysis.save();
      console.log('✅ Results saved to database');
    } catch (dbError) {
      console.log('⚠️  Could not save to database:', dbError.message);
    }

    res.json({
      success: true,
      data: {
        positive: result.positive,
        negative: result.negative,
        neutral: result.neutral,
        processing_time: `${result.processing_time}s`,
        total_processed: result.total_processed,
        method: result.method,
        model_used: result.model_used,
        avg_confidence: result.avg_confidence,
      },
    });

  } catch (error) {
    console.error('❌ Error in /api/analyze:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'Analysis failed',
      message: error.response?.data?.detail || error.message,
    });
  }
});

// Compare sequential vs parallel performance
app.post('/api/analyze/compare', async (req, res) => {
  try {
    const { texts } = req.body;

    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'Please provide an array of texts to analyze',
      });
    }

    console.log(`⚡ Running performance comparison on ${texts.length} texts...`);

    // Call Python service for comparison
    const response = await axios.post(`${PYTHON_SERVICE_URL}/analyze/compare`, {
      texts,
    }, {
      timeout: 120000, // 2 minute timeout for comparison
    });

    const result = response.data;

    // Save both results to MongoDB
    try {
      // Save sequential result
      await new Analysis({
        totalProcessed: result.total_processed,
        results: {
          positive: result.sequential.positive,
          negative: result.sequential.negative,
          neutral: result.sequential.neutral,
        },
        processingTime: result.sequential.processing_time,
        method: 'sequential',
      }).save();

      // Save parallel result
      await new Analysis({
        totalProcessed: result.total_processed,
        results: {
          positive: result.parallel.positive,
          negative: result.parallel.negative,
          neutral: result.parallel.neutral,
        },
        processingTime: result.parallel.processing_time,
        method: 'parallel',
        speedup: result.speedup,
        metadata: {
          numWorkers: result.parallel.num_workers,
          improvementPercent: result.improvement_percent,
        },
      }).save();

      console.log('✅ Comparison results saved to database');
    } catch (dbError) {
      console.log('⚠️  Could not save to database:', dbError.message);
    }

    res.json({
      success: true,
      data: result,
    });

  } catch (error) {
    console.error('❌ Error in /api/analyze/compare:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'Comparison failed',
      message: error.response?.data?.detail || error.message,
    });
  }
});

// Generate dataset
app.post('/api/generate-dataset', async (req, res) => {
  try {
    const { count = 10000, distribution } = req.body;

    console.log(`🎲 Generating dataset with ${count} records...`);

    // Call Python service
    const response = await axios.post(`${PYTHON_SERVICE_URL}/generate-dataset`, {
      count,
      distribution,
    });

    res.json({
      success: true,
      data: response.data,
    });

  } catch (error) {
    console.error('❌ Error in /api/generate-dataset:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'Dataset generation failed',
      message: error.response?.data?.detail || error.message,
    });
  }
});

// Get analysis history
app.get('/api/results', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    
    const results = await Analysis.find()
      .sort({ timestamp: -1 })
      .limit(limit);

    res.json({
      success: true,
      data: results,
    });

  } catch (error) {
    console.error('❌ Error in /api/results:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'Failed to fetch results',
      message: error.message,
    });
  }
});

// Get statistics
app.get('/api/stats', async (req, res) => {
  try {
    const totalAnalyses = await Analysis.countDocuments();
    const recentAnalyses = await Analysis.find().sort({ timestamp: -1 }).limit(5);
    
    const totalProcessed = recentAnalyses.reduce((sum, a) => sum + a.totalProcessed, 0);
    const avgProcessingTime = recentAnalyses.reduce((sum, a) => sum + a.processingTime, 0) / recentAnalyses.length;

    res.json({
      success: true,
      data: {
        totalAnalyses,
        totalProcessed,
        avgProcessingTime: avgProcessingTime.toFixed(3),
        recentAnalyses,
      },
    });

  } catch (error) {
    console.error('❌ Error in /api/stats:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'Failed to fetch statistics',
      message: error.message,
    });
  }
});

// NEW: Upload file for batch analysis
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded',
      });
    }

    const { textColumn = 'text', model = 'vader' } = req.body;

    console.log(`📁 Processing file: ${req.file.originalname} (${req.file.size} bytes) with ${model} model...`);

    // Create FormData to send to Python service
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    formData.append('text_column', textColumn);
    formData.append('model', model);

    // Call Python service
    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/upload-file`,
      formData,
      {
        headers: formData.getHeaders(),
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 300000, // 5 minute timeout for large files
      }
    );

    const result = response.data;

    // Save to MongoDB
    try {
      await new Analysis({
        totalProcessed: result.total_processed,
        results: {
          positive: result.summary.positive,
          negative: result.summary.negative,
          neutral: result.summary.neutral,
        },
        processingTime: result.processing_time,
        method: result.method,
        metadata: {
          fileName: req.file.originalname,
          fileType: result.file_type,
          model: model,
        },
      }).save();
    } catch (dbError) {
      console.log('⚠️  Could not save to database:', dbError.message);
    }

    res.json({
      success: true,
      data: result,
    });

  } catch (error) {
    console.error('❌ Error in /api/upload:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'File processing failed',
      message: error.response?.data?.detail || error.message,
    });
  }
});

// NEW: Trend analysis
app.post('/api/trend-analysis', async (req, res) => {
  try {
    const { texts, timestamps, interval = 'hour', model = 'vader' } = req.body;

    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'Please provide an array of texts to analyze',
      });
    }

    console.log(`📈 Running trend analysis on ${texts.length} texts with ${interval} interval...`);

    // Call Python service
    const response = await axios.post(`${PYTHON_SERVICE_URL}/trend-analysis`, {
      texts,
      timestamps,
      interval,
      model,
    }, {
      timeout: 180000, // 3 minute timeout
    });

    res.json({
      success: true,
      data: response.data,
    });

  } catch (error) {
    console.error('❌ Error in /api/trend-analysis:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'Trend analysis failed',
      message: error.response?.data?.detail || error.message,
    });
  }
});

// NEW: Get available models
app.get('/api/models', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_SERVICE_URL}/models`);
    
    res.json({
      success: true,
      data: response.data,
    });

  } catch (error) {
    console.error('❌ Error in /api/models:', error.message);
    
    res.status(500).json({
      success: false,
      error: 'Failed to fetch models',
      message: error.message,
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log('\n' + '='.repeat(60));
  console.log('🚀 Advanced Sentiment Analysis Backend Server');
  console.log('='.repeat(60));
  console.log(`📡 Server running on: http://localhost:${PORT}`);
  console.log(`🐍 Python service: ${PYTHON_SERVICE_URL}`);
  console.log(`💾 MongoDB: ${MONGODB_URI}`);
  console.log('='.repeat(60));
  console.log('📋 Available Endpoints:');
  console.log('   POST /api/analyze - Sentiment analysis');
  console.log('   POST /api/analyze/compare - Performance comparison');
  console.log('   POST /api/upload - Batch file processing');
  console.log('   POST /api/trend-analysis - Trend analysis');
  console.log('   POST /api/generate-dataset - Generate test data');
  console.log('   GET  /api/models - Available ML models');
  console.log('   GET  /api/results - Analysis history');
  console.log('   GET  /api/stats - Statistics');
  console.log('   GET  /api/health - Health check');
  console.log('='.repeat(60) + '\n');
});

module.exports = app;
