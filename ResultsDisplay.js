import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import AssessmentIcon from '@mui/icons-material/Assessment';

const ResultsDisplay = ({ results }) => {
  const data = results.rankedResumes.map((resume, index) => ({
    name: resume.name,
    score: parseFloat(resume.score),
    rank: index + 1
  }));

  return (
    <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <AssessmentIcon color="primary" />
        <Typography variant="h5" component="h2">
          Ranking Results
        </Typography>
      </Box>

      <Box sx={{ height: 400, width: '100%' }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 60
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              angle={-45}
              textAnchor="end"
              height={60}
              interval={0}
            />
            <YAxis 
              domain={[0, 1]}
              label={{ 
                value: 'Relevance Score', 
                angle: -90, 
                position: 'insideLeft' 
              }}
            />
            <Tooltip />
            <Bar 
              dataKey="score" 
              fill="#1976d2"
              name="Relevance Score"
            />
          </BarChart>
        </ResponsiveContainer>
      </Box>

      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Detailed Rankings:
        </Typography>
        {data.map((item, index) => (
          <Box 
            key={index}
            sx={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              alignItems: 'center',
              p: 1,
              borderBottom: index < data.length - 1 ? '1px solid #eee' : 'none'
            }}
          >
            <Typography>
              {index + 1}. {item.name}
            </Typography>
            <Typography color="primary">
              Score: {(item.score * 100).toFixed(1)}%
            </Typography>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default ResultsDisplay; 