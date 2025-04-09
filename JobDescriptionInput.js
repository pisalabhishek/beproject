import React from 'react';
import { Paper, Typography, TextField, Box } from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';

const JobDescriptionInput = ({ value, onChange }) => {
  return (
    <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <DescriptionIcon color="primary" />
        <Typography variant="h6" component="h2">
          Job Description
        </Typography>
      </Box>
      <TextField
        fullWidth
        multiline
        rows={8}
        variant="outlined"
        placeholder="Enter the job description here..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        sx={{
          '& .MuiOutlinedInput-root': {
            '&:hover fieldset': {
              borderColor: 'primary.main',
            },
          },
        }}
      />
    </Paper>
  );
};

export default JobDescriptionInput; 