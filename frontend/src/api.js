import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const analyzeSentiment = async (texts, parallel = true, model = 'vader', numWorkers = null) => {
  try {
    const payload = {
      texts,
      parallel,
      model,
    };
    if (numWorkers !== null) {
      payload.num_workers = numWorkers;
    }
    const response = await axios.post(`${API_BASE_URL}/analyze`, payload);
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

export const comparePerformance = async (texts) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/analyze/compare`, {
      texts,
    });
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

export const generateDataset = async (count = 10000, distribution = null) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/generate-dataset`, {
      count,
      distribution,
    });
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

// NEW: Upload file for batch processing
export const uploadFile = async (file, textColumn = 'text', model = 'vader') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('textColumn', textColumn);
    formData.append('model', model);

    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

// NEW: Trend analysis
export const analyzeTrend = async (texts, timestamps = null, interval = 'hour', model = 'vader') => {
  try {
    const response = await axios.post(`${API_BASE_URL}/trend-analysis`, {
      texts,
      timestamps,
      interval,
      model,
    });
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

// NEW: Get available models
export const getAvailableModels = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/models`);
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

export const getResults = async (limit = 10) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/results?limit=${limit}`);
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

export const getStats = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/stats`);
    return response.data;
  } catch (error) {
    throw error.response?.data || error;
  }
};
