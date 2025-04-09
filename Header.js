import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import WorkIcon from '@mui/icons-material/Work';

const Header = () => {
  return (
    <AppBar position="static" color="primary" elevation={0}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <WorkIcon sx={{ fontSize: 28 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Resume Ranker
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 