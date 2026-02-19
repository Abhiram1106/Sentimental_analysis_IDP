import React from 'react';
import { Bar } from 'react-chartjs-2';

const ComparisonChart = ({ comparisonData }) => {
  const { sequential, parallel, speedup, improvement_percent, recommendation } = comparisonData;

  const data = {
    labels: ['🐢 Sequential', '⚡ Parallel'],
    datasets: [
      {
        label: 'Processing Time (seconds)',
        data: [sequential.processing_time, parallel.processing_time],
        backgroundColor: [
          'rgba(245, 158, 11, 0.85)',
          'rgba(16, 185, 129, 0.85)',
        ],
        borderColor: [
          'rgba(245, 158, 11, 1)',
          'rgba(16, 185, 129, 1)',
        ],
        borderWidth: 2,
        borderRadius: 10,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Sequential vs Parallel Performance',
        font: {
          size: 16,
          weight: '700',
          family: 'Inter',
        },
        padding: {
          bottom: 20,
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
        callbacks: {
          label: function(context) {
            return `Time: ${context.parsed.y}s`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        title: {
          display: true,
          text: 'Time (seconds)',
          font: {
            size: 13,
            weight: '600',
          },
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
    <div className="comparison-container">
      <h2>⚡ Performance Comparison</h2>
      
      {recommendation && (
        <div className={`info-banner ${
          recommendation.includes('✅') ? 'success-banner' : 'warning-banner'
        }`} style={{
          padding: '1rem 1.5rem',
          borderRadius: '0.75rem',
          marginBottom: '1.5rem',
          fontSize: '1rem',
          fontWeight: '600',
          textAlign: 'center',
          background: recommendation.includes('✅') 
            ? 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)'
            : 'linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)',
          color: '#2d3436',
          border: `2px solid ${recommendation.includes('✅') ? '#28a745' : '#ffc107'}`,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          {recommendation}
        </div>
      )}
      
      <div className="comparison-stats">
        <div className="comparison-stat-card">
          <h3>Speedup Achieved</h3>
          <p className="stat-value speedup">{speedup}x</p>
        </div>
        <div className="comparison-stat-card">
          <h3>Performance Boost</h3>
          <p className="stat-value improvement">{improvement_percent.toFixed(1)}%</p>
        </div>
        <div className="comparison-stat-card">
          <h3>Sequential Time</h3>
          <p className="stat-value">{sequential.processing_time}s</p>
        </div>
        <div className="comparison-stat-card">
          <h3>Parallel Time</h3>
          <p className="stat-value success">{parallel.processing_time}s</p>
        </div>
      </div>

      <div className="chart-wrapper" style={{ marginTop: '2rem' }}>
        <Bar data={data} options={options} />
      </div>

      <div className="comparison-stats" style={{ marginTop: '2rem', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
        <div className="comparison-stat-card" style={{ 
          background: 'linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%)',
          color: '#2d3436'
        }}>
          <h3 style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>🐢 Sequential Processing</h3>
          <div style={{ display: 'grid', gap: '0.75rem', textAlign: 'left' }}>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>⏱️ Processing Time:</span>
              <strong>{sequential.processing_time}s</strong>
            </p>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>😊 Positive:</span>
              <strong>{sequential.positive}</strong>
            </p>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>😞 Negative:</span>
              <strong>{sequential.negative}</strong>
            </p>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>😐 Neutral:</span>
              <strong>{sequential.neutral}</strong>
            </p>
            <p style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              fontWeight: '800',
              fontSize: '1.1rem',
              paddingTop: '0.75rem',
              borderTop: '2px solid rgba(0,0,0,0.1)'
            }}>
              <span>📊 Total:</span>
              <strong>{sequential.positive + sequential.negative + sequential.neutral}</strong>
            </p>
          </div>
        </div>

        <div className="comparison-stat-card" style={{ 
          background: 'linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%)',
          color: '#2d3436'
        }}>
          <h3 style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>⚡ Parallel Processing</h3>
          <div style={{ display: 'grid', gap: '0.75rem', textAlign: 'left' }}>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>⏱️ Processing Time:</span>
              <strong>{parallel.processing_time}s</strong>
            </p>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>👷 CPU Workers:</span>
              <strong>{parallel.num_workers}</strong>
            </p>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>😊 Positive:</span>
              <strong>{parallel.positive}</strong>
            </p>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>😞 Negative:</span>
              <strong>{parallel.negative}</strong>
            </p>
            <p style={{ display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
              <span>😐 Neutral:</span>
              <strong>{parallel.neutral}</strong>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonChart;
