import { useEffect, useState } from "react";
import { DataGrid } from "@mui/x-data-grid";
import { fetchHitl, saveHitl } from "../api/inventory_api";

export default function InventoryTable() {
  const [rows, setRows] = useState([]);
  const [search, setSearch] = useState("");

  useEffect(() => {
    fetchHitl().then((res) => setRows(res.data)); // same API can return all records
  }, []);

  const filtered = rows.filter(
    (r) =>
      r.description?.toLowerCase().includes(search.toLowerCase()) ||
      r.brand?.toLowerCase().includes(search.toLowerCase()) ||
      r.mapped_category?.toLowerCase().includes(search.toLowerCase())
  );

  const handleSave = async () => {
    await saveHitl(rows);
    alert("✅ All changes saved successfully!");
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-4">
        <input
          type="text"
          placeholder="Search by keyword..."
          className="border p-2 rounded w-1/3"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <button
          onClick={handleSave}
          className="bg-indigo-600 text-white px-5 py-2 rounded hover:bg-indigo-700"
        >
          Save All
        </button>
      </div>

      <div style={{ height: 550, width: "100%" }}>
        <DataGrid
          rows={filtered}
          columns={[
            { field: "id", headerName: "ID", width: 80 },
            { field: "sku", headerName: "SKU", width: 150, editable: true },
            { field: "description", headerName: "Description", width: 300, editable: true },
            { field: "brand", headerName: "Brand", width: 180, editable: true },
            { field: "mapped_category", headerName: "Category", width: 180, editable: true },
            { field: "confidence", headerName: "Confidence", width: 150 },
          ]}
          pageSize={10}
          onCellEditStop={(params) => {
            setRows((prev) =>
              prev.map((r) =>
                r.id === params.id ? { ...r, [params.field]: params.value } : r
              )
            );
          }}
        />
      </div>
    </div>
  );
}
