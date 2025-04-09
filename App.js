import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Paper, 
  ThemeProvider, 
  createTheme,
  CssBaseline
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import ResumeUploader from './components/ResumeUploader';
import JobDescriptionInput from './components/JobDescriptionInput';
import ResultsDisplay from './components/ResultsDisplay';
import Header from './components/Header';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
});

function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [resumes, setResumes] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const onDrop = (acceptedFiles) => {
    setResumes(acceptedFiles);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    }
  });

  const handleSubmit = async () => {
    if (!jobDescription || resumes.length === 0) {
      alert('Please provide both job description and resumes');
      return;
    }

    setLoading(true);
    // Here you would make the API call to your backend
    // For now, we'll just simulate a delay
    setTimeout(() => {
      setLoading(false);
      // Mock results - replace with actual API response
      setResults({
        rankedResumes: resumes.map((file, index) => ({
          name: file.name,
          score: (0.8 - index * 0.1).toFixed(2)
        }))
      });
    }, 2000);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        <Header />
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
            <Typography variant="h1" gutterBottom align="center">
              AI-Powered Resume Ranking
            </Typography>
            <Typography variant="h6" color="text.secondary" align="center" gutterBottom>
              Upload resumes and enter a job description to rank candidates based on relevance
            </Typography>
          </Paper>

          <Box sx={{ display: 'grid', gap: 4, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
            <JobDescriptionInput 
              value={jobDescription}
              onChange={setJobDescription}
            />
            
            <ResumeUploader 
              getRootProps={getRootProps}
              getInputProps={getInputProps}
              isDragActive={isDragActive}
              resumes={resumes}
            />
          </Box>

          <Box sx={{ mt: 4, textAlign: 'center' }}>
            <button
              onClick={handleSubmit}
              disabled={loading || !jobDescription || resumes.length === 0}
              style={{
                padding: '12px 24px',
                fontSize: '1.1rem',
                backgroundColor: theme.palette.primary.main,
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.7 : 1
              }}
            >
              {loading ? 'Processing...' : 'Rank Resumes'}
            </button>
          </Box>

          {results && (
            <ResultsDisplay results={results} />
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App; 