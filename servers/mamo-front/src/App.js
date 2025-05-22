import React, { useState } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import logo from './logo.svg';
import './App.css';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const config = require('./config.json');

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setOriginalImage(URL.createObjectURL(file));
    setIsLoading(true);
    uploadImage(file);
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  const uploadImage = (file) => {
    const formData = new FormData();
    formData.append('file', file);

    axios.post(`${config.server_url}`, formData, {
      responseType: 'blob',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    .then(response => {
      const url = URL.createObjectURL(new Blob([response.data]));
      setProcessedImage(url);
      setIsLoading(false);
    })
    .catch(error => {
      console.error('Error uploading image:', error);
      setIsLoading(false);
    });
  };

  return (
    <div className="App">
      <header className="app-header">
        <img src={logo} alt="Logo" className="logo" />
        <h4 className="app-title">Истраживачко-развојни институт за вештачку интелигенцију Србије</h4>
        <h1 className="app-title">Проналажење сумњивих маса на мамограму</h1>
      </header>
      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        <p>Превући слику овде</p>
      </div>
      <div className="image-preview">
        {isLoading ? (
          <div className="loading-spinner"></div>
        ) : (
          <>
            {originalImage && <img src={originalImage} alt="Original" className="image" />}
            {processedImage && <img src={processedImage} alt="Processed" className="image" />}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
