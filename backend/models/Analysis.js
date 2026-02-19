const mongoose = require('mongoose');

const analysisSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    default: Date.now,
  },
  totalProcessed: {
    type: Number,
    required: true,
  },
  results: {
    positive: {
      type: Number,
      required: true,
    },
    negative: {
      type: Number,
      required: true,
    },
    neutral: {
      type: Number,
      required: true,
    },
  },
  processingTime: {
    type: Number,
    required: true,
  },
  method: {
    type: String,
    enum: ['sequential', 'parallel'],
    required: true,
  },
  speedup: {
    type: Number,
    default: null,
  },
  metadata: {
    numWorkers: Number,
    improvementPercent: Number,
  },
});

module.exports = mongoose.model('Analysis', analysisSchema);
