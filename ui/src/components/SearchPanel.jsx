import { useState } from "react";
import { searchInventory, downloadFinal } from "../api/inventory_api";

export default function SearchPanel() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    const res = await searchInventory(query);
    setResults(res.data);
  };

  const handleDownload = async () => {
    const res = await downloadFinal();
    const blob = new Blob([res.data]);
    const link = document.createElement("a");
    link.href = window.URL.createObjectURL(blob);
    link.download = "final_inventory.xlsx";
    link.click();
  };

  return (
    <div className="p-10 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">📦 Final Inventory Records</h1>
      <div className="flex gap-4 mb-6">
        <input
          type="text"
          placeholder="Search SKU / Brand / Category..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="border px-4 py-2 rounded w-1/2"
        />
        <button onClick={handleSearch} className="bg-blue-600 text-white px-4 py-2 rounded">
          Search
        </button>
        <button onClick={handleDownload} className="bg-green-600 text-white px-4 py-2 rounded">
          Download
        </button>
      </div>

      <table className="w-full border">
        <thead>
          <tr className="bg-gray-100 text-left">
            <th className="p-2 border">SKU</th>
            <th className="p-2 border">Description</th>
            <th className="p-2 border">Brand</th>
            <th className="p-2 border">Category</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} className="border-t">
              <td className="p-2">{r.sku}</td>
              <td className="p-2">{r.description}</td>
              <td className="p-2">{r.brand}</td>
              <td className="p-2">{r.mapped_category}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
