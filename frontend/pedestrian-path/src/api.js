import axios from 'axios';

export const uploadMapData = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post('/api/upload', formData);
  return response.data;
};