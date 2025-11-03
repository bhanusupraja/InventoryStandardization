import { useState } from "react";
import { uploadFile } from "../api/inventory_api";

export default function FileUpload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Select a file first!");
    setLoading(true);
    await uploadFile(file);
    setLoading(false);
    alert("✅ File uploaded successfully!");
  };

  return (
    <div className="p-10 min-h-screen bg-gray-50">
      <h1 className="text-3xl font-bold mb-6">📂 Upload Inventory Data</h1>
      <input
        type="file"
        accept=".csv,.xlsx,.xls,.json"
        onChange={(e) => setFile(e.target.files[0])}
        className="border p-2 rounded mb-4 w-1/2"
      />
      <button
        onClick={handleUpload}
        disabled={loading}
        className="bg-blue-600 text-white px-5 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? "Uploading..." : "Upload & Process"}
      </button>
    </div>
  );
}
