# 🚀 Advanced Sentiment Analysis Platform

A production-ready, full-stack machine learning application for sentiment analysis featuring parallel processing, dual ML models (VADER & DistilBERT), batch file processing, and time-series trend analysis.

---

## ✨ Key Features

### 🤖 Dual ML Models
- **VADER**: Fast rule-based analysis for real-time processing
- **DistilBERT**: Advanced transformer model for high-accuracy sentiment detection
- Dynamic model switching with automatic availability detection

### ⚡ High-Performance Processing
- **Parallel Processing Engine**: 4-5x speedup utilizing all CPU cores
- **Batch File Processing**: Upload and analyze CSV, TXT, XLSX files (up to 50MB)
- **Smart Dataset Generation**: Create synthetic test data with customizable distributions

### 📈 Trend Analysis
- Time-series sentiment tracking with trend direction detection
- Moving averages, volatility analysis, and 3-period predictive modeling
- Interactive visualizations with Chart.js

### 🎨 Modern UI/UX
- Responsive single-page application with glassmorphism design
- Tab-based navigation with smooth animations
- Real-time feedback and performance metrics
- Cross-device compatibility

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          Frontend (React 18.2)              │
│    File Upload • Charts • Model Selector    │
└────────────────┬────────────────────────────┘
                 │ REST API
┌────────────────▼────────────────────────────┐
│       Backend (Node.js + Express)           │
│    API Gateway • MongoDB • File Handling    │
└────────────────┬────────────────────────────┘
                 │ REST API
┌────────────────▼────────────────────────────┐
│      Python Engine (FastAPI)                │
│  • VADER Analyzer                           │
│  • DistilBERT Transformer                   │
│  • Batch Processor (CSV/TXT/XLSX)           │
│  • Trend Analyzer (Time-Series)             │
└─────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
IDP-V/
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── App.js          # Main application component
│   │   └── api.js          # API service layer
│   ├── public/
│   └── package.json
│
├── backend/                # Node.js + Express server
│   ├── models/
│   │   └── Analysis.js     # MongoDB schema
│   ├── server.js           # Main server file
│   └── package.json
│
├── python-engine/          # Python ML engine
│   ├── main.py             # FastAPI application
│   ├── sentiment_analyzer.py    # VADER implementation
│   ├── transformer_analyzer.py  # DistilBERT implementation
│   ├── batch_processor.py       # File processing
│   ├── trend_analyzer.py        # Trend analysis
│   └── requirements.txt
│
├── IMDB-Dataset.csv        # Sample dataset
└── README.md               # Project documentation
```

---

## 📦 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18.2, Chart.js 4.4, Axios, CSS3 Glassmorphism |
| **Backend** | Node.js, Express 4.18, MongoDB, Mongoose 8.0, Multer |
| **ML Engine** | FastAPI, HuggingFace Transformers, PyTorch, VADER, scikit-learn |
| **Data Processing** | Pandas, NumPy, SciPy |

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 16+ 
- **Python** 3.8+
- **MongoDB** (optional - for data persistence)
- **RAM**: 4GB+ recommended (for transformer model)

### Installation & Setup

#### 1️⃣ Python ML Engine
```bash
cd python-engine
pip install -r requirements.txt
python main.py
```
**Server running at:** `http://localhost:8000`

#### 2️⃣ Backend API Server
```bash
cd backend
npm install
npm start
```
**Server running at:** `http://localhost:5000`

#### 3️⃣ Frontend Application
```bash
cd frontend
npm install
npm start
```
**Application opens at:** `http://localhost:3000`

> **Note:** All three services must be running simultaneously for the application to work.

---

## 📖 Usage Guide

### 📝 Text Analysis
1. Select your preferred model (VADER for speed, DistilBERT for accuracy)
2. Choose input method:
   - **Paste Text**: Enter multiple texts (one per line)
   - **Generate Dataset**: Create synthetic test data (100-50,000 texts)
   - **Manual Entry**: Type or paste individual texts
3. Click **Analyze** to process
4. View results with sentiment distribution charts and performance metrics

### 📁 File Upload & Batch Processing
1. **Upload File**: Drag & drop or select file (CSV, TXT, XLSX - max 50MB)
2. **Configure**:
   - For structured data (CSV/Excel): Specify text column name
   - Select analysis model
3. **Process**: Click "Analyze File"
4. **Download Results**: Export analyzed data with sentiment scores

### 📈 Trend Analysis
1. **Input Time-Series Data**: Enter texts with timestamps
2. **Select Model**: Choose VADER or DistilBERT
3. **Analyze**: Click "Analyze Trends"
4. **View Insights**:
   - Trend direction (improving ↗️ / declining ↘️ / stable →)
   - Sentiment over time graphs
   - Volatility metrics and predictions

---

## 🔌 API Endpoints

### Sentiment Analysis
**POST** `/api/analyze`
```json
{
  "texts": ["I love this!", "This is bad"],
  "parallel": true,
  "model": "vader"
}
```

### Performance Comparison
**POST** `/api/analyze/compare` - Compare sequential vs parallel processing

### Batch File Upload
**POST** `/api/upload` - Upload CSV/TXT/XLSX for batch analysis

### Trend Analysis
**POST** `/api/trend-analysis` - Analyze sentiment trends over time

### Utilities
- **POST** `/api/generate-dataset` - Generate synthetic test data
- **GET** `/api/models` - List available ML models and their status
- **GET** `/api/results` - Retrieve analysis history (MongoDB)
- **GET** `/api/stats` - Get application statistics

---

## ⚡ Performance Benchmarks

| Model | Speed (10K texts) | Accuracy | Memory | Best For |
|-------|------------------|----------|---------|----------|
| **VADER** | ~2-3 sec (parallel) | Good | ~100MB | Large datasets, real-time |
| **DistilBERT** | ~15-30 sec (parallel) | Excellent (90%+) | ~1GB | Quality analysis, complex text |

**Parallel Processing Benefits:**
- 4-5x speedup on 8-core CPU
- Linear scaling with CPU core count
- Optimal for datasets >1,000 texts

---

## 🎨 UI Features

- **Modern Design**: Glassmorphism with gradient backgrounds (purple → violet → pink)
- **Responsive Layout**: Optimized for desktop, tablet, and mobile devices
- **Interactive Charts**: Real-time sentiment visualizations with Chart.js
- **Smooth Animations**: Transitions and hover effects for enhanced UX
- **Accessibility**: Keyboard navigation, high contrast, reduced motion support

---

## 📊 Supported File Formats

| Format | Features |
|--------|----------|
| **TXT** | One text per line, UTF-8 encoding, auto line-break detection |
| **CSV** | Custom column selection, headers supported, UTF-8 encoding |
| **XLSX** | Multiple sheets, column selection, headers required |

**Processing Limits:**
- Max file size: 50MB
- Recommended: <100,000 rows for optimal performance
- Processing timeout: 5 minutes

---

## � Trend Analysis Capabilities

**Metrics Calculated:**
- Trend direction (mathematical slope analysis)
- Correlation strength (weak/moderate/strong)
- Volatility (standard deviation)
- Moving averages (smoothed trend lines)
- Peak period detection

**Prediction Model:**
- Linear regression-based forecasting
- 3-period predictions with confidence scores
- Adaptive time intervals (minute/hour/day/week)

---

## ⚙️ Configuration

### Environment Variables

**Python Engine** (`python-engine/.env`)
```env
PORT=8000
DEBUG=True
MODEL_CACHE_DIR=./model_cache
```

**Backend** (`backend/.env`)
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/sentiment_analysis
PYTHON_SERVICE_URL=http://localhost:8000
```

**Frontend** (`frontend/.env`)
```env
REACT_APP_API_URL=http://localhost:5000/api
```

---

## 🔧 Troubleshooting

### Transformer Model Issues
```bash
pip install transformers torch accelerate
# Clear cache if needed
rm -rf ~/.cache/huggingface/transformers
```

### File Upload Errors
- Verify file size is under 50MB
- Check file format (CSV/TXT/XLSX only)
- Ensure correct text column name for CSV/Excel
- Review backend logs for detailed error messages

### MongoDB Connection
- Application works without MongoDB (results won't persist)
- Verify MongoDB is running: `mongod --version`
- Check connection string in `backend/.env`

### Frontend Build Issues
```bash
rm -rf node_modules package-lock.json
npm install
npm start
```

---

## 🎯 Future Enhancements

- [ ] Real-time streaming analysis
- [ ] Multi-language sentiment support
- [ ] Custom model fine-tuning
- [ ] Export reports (PDF/Excel)
- [ ] API authentication & rate limiting
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Social media integration
- [ ] Advanced NLP features (entity extraction, topic modeling)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - feel free to use it in your own projects!

---

## 🙏 Acknowledgments

- **HuggingFace** - Transformer models and NLP tools
- **VADER Sentiment** - Lexicon-based sentiment analysis
- **FastAPI** - High-performance Python web framework
- **React** - Modern frontend framework
- **Chart.js** - Beautiful data visualizations

---

## 📞 Support & Contact

For issues, questions, or feature requests:
- Check the documentation above
- Review logs for error details
- Ensure all services are running
- Verify dependencies are properly installed

---

**Built with ❤️ for Advanced Sentiment Analysis**

🚀 **Version**: 2.0  
📅 **Updated**: February 2026  
⭐ **Status**: Production Ready
