import React from 'react';
import { Pie, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const SentimentChart = ({ results }) => {
  const pieData = {
    labels: ['😊 Positive', '😞 Negative', '😐 Neutral'],
    datasets: [
      {
        label: 'Sentiment Distribution',
        data: [results.positive, results.negative, results.neutral],
        backgroundColor: [
          'rgba(16, 185, 129, 0.85)',
          'rgba(239, 68, 68, 0.85)',
          'rgba(245, 158, 11, 0.85)',
        ],
        borderColor: [
          'rgba(16, 185, 129, 1)',
          'rgba(239, 68, 68, 1)',
          'rgba(245, 158, 11, 1)',
        ],
        borderWidth: 3,
        hoverOffset: 15,
      },
    ],
  };

  const barData = {
    labels: ['😊 Positive', '😞 Negative', '😐 Neutral'],
    datasets: [
      {
        label: 'Number of Messages',
        data: [results.positive, results.negative, results.neutral],
        backgroundColor: [
          'rgba(16, 185, 129, 0.85)',
          'rgba(239, 68, 68, 0.85)',
          'rgba(245, 158, 11, 0.85)',
        ],
        borderColor: [
          'rgba(16, 185, 129, 1)',
          'rgba(239, 68, 68, 1)',
          'rgba(245, 158, 11, 1)',
        ],
        borderWidth: 2,
        borderRadius: 8,
      },
    ],
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 20,
          font: {
            size: 13,
            weight: '600',
            family: 'Inter',
          },
          usePointStyle: true,
          pointStyle: 'circle',
        },
      },
      tooltip: {
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        padding: 12,
        titleFont: {
          size: 14,
          weight: '600',
        },
        bodyFont: {
          size: 13,
        },
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
      },
    },
    animation: {
      animateRotate: true,
      animateScale: true,
    },
  };

  const barOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        padding: 12,
        titleFont: {
          size: 14,
          weight: '600',
        },
        bodyFont: {
          size: 13,
        },
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          font: {
            size: 12,
            weight: '500',
          },
        },
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          font: {
            size: 13,
            weight: '600',
          },
        },
      },
    },
  };

  return (
    <div className="charts-container">
      <div className="chart-wrapper">
        <h3>🥧 Sentiment Distribution</h3>
        <div style={{ maxHeight: '400px', position: 'relative' }}>
          <Pie data={pieData} options={pieOptions} />
        </div>
      </div>

      <div className="chart-wrapper">
        <h3>📊 Breakdown Analysis</h3>
        <div style={{ maxHeight: '400px', position: 'relative' }}>
          <Bar data={barData} options={barOptions} />
        </div>
      </div>
    </div>
  );
};

export default SentimentChart;
