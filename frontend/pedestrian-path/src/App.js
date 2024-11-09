import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [imageSrc, setImageSrc] = useState(null);
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleClickUpload = () => {
    fileInputRef.current.click();
  };

  const processFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      const coordinates = content
        .split(',')
        .map(pair => pair.trim().split(' ').map(Number));

      sendCoordinatesToServer(coordinates);
    };
    reader.readAsText(file);
  };

  const sendCoordinatesToServer = async (coordinates) => {
    try {
      const response = await fetch('http://localhost:5000/generate_map', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          coordinates: coordinates,
          color: 'blue'
        }),
      });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setImageSrc(url);
    } catch (error) {
      console.error('Error generating map image:', error);
    }
  };

  return (
    <div className="App">
      <div className="sidebar">
        <h1>PedestrianPath</h1>
      </div>
      <div className="content">
        <div
          className={`upload-container ${isDragging ? 'dragging' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onClick={handleClickUpload}
          style={{ border: '2px dashed blue', padding: '20px', textAlign: 'center', cursor: 'pointer', width: '100%', marginBottom: '20px' }}
        >
          <input type="file" ref={fileInputRef} style={{ display: 'none' }} onChange={handleFileUpload} />
          <p>Drag & Drop (Загрузка карты)</p>
        </div>

        {imageSrc && (
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <img src={imageSrc} alt="Generated Map" style={{ width: '80%', height: 'auto' }} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
