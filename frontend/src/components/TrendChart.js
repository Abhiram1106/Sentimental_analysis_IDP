import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function TrendChart({ trendData }) {
  if (!trendData || !trendData.time_series || trendData.time_series.length === 0) {
    return <div className="no-data">No trend data available</div>;
  }

  const timeSeries = trendData.time_series;

  // Extract timestamps and data
  const labels = timeSeries.map(point => {
    const date = new Date(point.timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  });

  const positiveData = timeSeries.map(point => point.positive);
  const negativeData = timeSeries.map(point => point.negative);
  const neutralData = timeSeries.map(point => point.neutral);
  const compoundData = timeSeries.map(point => point.avg_compound);

  const sentimentCountData = {
    labels,
    datasets: [
      {
        label: 'Positive 😊',
        data: positiveData,
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Negative 😞',
        data: negativeData,
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Neutral 😐',
        data: neutralData,
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const compoundTrendData = {
    labels,
    datasets: [
      {
        label: 'Average Compound Score',
        data: compoundData,
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        fill: true,
        tension: 0.4,
        borderWidth: 3,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: {
            family: 'Inter, sans-serif',
            size: 13,
            weight: 600,
          },
          color: '#4a5568',
          usePointStyle: true,
          padding: 18,
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        titleColor: '#ffffff',
        bodyColor: '#e5e7eb',
        borderColor: '#6366f1',
        borderWidth: 2,
        padding: 16,
        titleFont: {
          family: 'Inter, sans-serif',
          size: 14,
          weight: 700,
        },
        bodyFont: {
          family: 'Inter, sans-serif',
          size: 13,
          weight: 500,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.06)',
        },
        ticks: {
          color: '#64748b',
          font: {
            family: 'Inter, sans-serif',
            size: 12,
            weight: 500,
          },
        },
      },
      x: {
        grid: {
          color: 'rgba(0, 0, 0, 0.04)',
        },
        ticks: {
          color: '#64748b',
          font: {
            family: 'Inter, sans-serif',
            size: 11,
            weight: 500,
          },
          maxRotation: 45,
          minRotation: 45,
        },
      },
    },
  };

  return (
    <div>
      <div className="chart-wrapper" style={{ marginBottom: '2rem' }}>
        <h3>📈 Sentiment Counts Over Time</h3>
        <div style={{ height: '400px', position: 'relative' }}>
          <Line data={sentimentCountData} options={options} />
        </div>
      </div>

      <div className="chart-wrapper">
        <h3>📊 Compound Score Trend</h3>
        <div style={{ height: '400px', position: 'relative' }}>
          <Line data={compoundTrendData} options={{
            ...options,
            scales: {
              ...options.scales,
              y: {
                ...options.scales.y,
                min: -1,
                max: 1,
              },
            },
          }} />
        </div>
      </div>
    </div>
  );
}

export default TrendChart;
