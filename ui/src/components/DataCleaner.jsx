import { useState } from "react";
import { preprocessData } from "../api/inventory_api";
import ConfidenceVisualizer from "./ConfidenceVisualizer";

export default function DataCleaner() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleClean = async () => {
    setLoading(true);
    const res = await preprocessData();
    setStats(res.data.confidence_stats);
    setLoading(false);
  };

  return (
    <div className="p-10 min-h-screen bg-gray-50">
      <h1 className="text-3xl font-bold mb-4">🧹 Data Preprocessing</h1>
      <button
        className="bg-green-600 text-white px-5 py-2 rounded hover:bg-green-700"
        onClick={handleClean}
        disabled={loading}
      >
        {loading ? "Processing..." : "Run Preprocessor"}
      </button>
      {stats && <ConfidenceVisualizer stats={stats} />}
    </div>
  );
}
