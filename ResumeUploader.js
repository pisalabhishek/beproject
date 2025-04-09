import React from 'react';
import { Paper, Typography, Box, List, ListItem, ListItemIcon, ListItemText, IconButton } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';

const ResumeUploader = ({ getRootProps, getInputProps, isDragActive, resumes }) => {
  return (
    <Paper 
      elevation={2} 
      sx={{ 
        p: 3, 
        height: '100%',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <CloudUploadIcon color="primary" />
        <Typography variant="h6" component="h2">
          Upload Resumes
        </Typography>
      </Box>

      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          borderRadius: 1,
          p: 3,
          textAlign: 'center',
          cursor: 'pointer',
          mb: 2,
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'action.hover'
          }
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
        <Typography>
          {isDragActive
            ? "Drop the resumes here"
            : "Drag 'n' drop resumes here, or click to select files"}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          (PDF files only)
        </Typography>
      </Box>

      {resumes.length > 0 && (
        <List sx={{ flexGrow: 1, overflow: 'auto' }}>
          {resumes.map((file, index) => (
            <ListItem
              key={index}
              secondaryAction={
                <IconButton edge="end" aria-label="delete">
                  <DeleteIcon />
                </IconButton>
              }
            >
              <ListItemIcon>
                <InsertDriveFileIcon color="primary" />
              </ListItemIcon>
              <ListItemText 
                primary={file.name}
                secondary={`${(file.size / 1024 / 1024).toFixed(2)} MB`}
              />
            </ListItem>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default ResumeUploader; 