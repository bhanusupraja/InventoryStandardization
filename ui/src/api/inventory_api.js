import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000"; // FastAPI server

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return axios.post(`${BASE_URL}/upload`, formData);
};

export const fetchHitl = async () => axios.get(`${BASE_URL}/records`);
export const saveHitl = async (rows) => axios.post(`${BASE_URL}/save`, rows);
