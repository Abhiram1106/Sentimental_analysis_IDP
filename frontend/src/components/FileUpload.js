import React, { useState } from 'react';

function FileUpload({ onUpload, isLoading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [textColumn, setTextColumn] = useState('text');
  const [model, setModel] = useState('vader');
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (selectedFile && onUpload) {
      await onUpload(selectedFile, textColumn, model);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="file-upload-section">
      <h3>📁 Batch File Processing</h3>
      <p className="section-description">
        Upload CSV, TXT, or Excel files for bulk sentiment analysis
      </p>

      <form onSubmit={handleSubmit} className="file-upload-form">
        <div
          className={`file-drop-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-input"
            accept=".txt,.csv,.xlsx,.xls"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          
          {!selectedFile ? (
            <label htmlFor="file-input" className="file-drop-label">
              <div className="upload-icon">📤</div>
              <p className="upload-text">
                <strong>Click to upload</strong> or drag and drop
              </p>
              <p className="upload-hint">
                TXT, CSV, or XLSX (Max 50MB)
              </p>
            </label>
          ) : (
            <div className="selected-file-info">
              <div className="file-icon">📄</div>
              <div className="file-details">
                <p className="file-name">{selectedFile.name}</p>
                <p className="file-size">{formatFileSize(selectedFile.size)}</p>
              </div>
              <button
                type="button"
                className="remove-file-btn"
                onClick={() => setSelectedFile(null)}
                aria-label="Remove file"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {selectedFile && (selectedFile.name.endsWith('.csv') || selectedFile.name.endsWith('.xlsx')) && (
          <div className="form-group" style={{ marginTop: '1.5rem' }}>
            <label htmlFor="text-column">Text Column Name:</label>
            <input
              type="text"
              id="text-column"
              value={textColumn}
              onChange={(e) => setTextColumn(e.target.value)}
              placeholder="Enter column name (e.g., text, comment, review)"
            />
          </div>
        )}

        <div className="form-group">
          <label htmlFor="model-select">Analysis Model:</label>
          <select
            id="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{
              width: '100%',
              padding: '1rem 1.25rem',
              border: '2px solid #e5e7eb',
              borderRadius: '1rem',
              fontSize: '1rem',
              fontWeight: '600',
              background: 'white',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              fontFamily: 'inherit'
            }}
          >
            <option value="vader">⚡ VADER - Fast & Accurate</option>
            <option value="transformer">🔥 DistilBERT - Deep Learning Transformer</option>
          </select>
        </div>

        <button
          type="submit"
          className="btn-primary"
          disabled={!selectedFile || isLoading}
          style={{ width: '100%', marginTop: '1rem' }}
        >
          {isLoading ? (
            <>
              <span style={{ 
                display: 'inline-block',
                width: '18px',
                height: '18px',
                border: '3px solid rgba(255,255,255,0.3)',
                borderTop: '3px solid white',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></span>
              Processing File...
            </>
          ) : (
            <>
              <span>🚀</span>
              Analyze File
            </>
          )}
        </button>
      </form>
    </div>
  );
}

export default FileUpload;
