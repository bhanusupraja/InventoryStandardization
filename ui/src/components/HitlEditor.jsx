import { useState, useEffect } from "react";
import { DataGrid } from "@mui/x-data-grid";
import { fetchHitl, saveHitl } from "../api/inventory_api";

export default function HitlEditor() {
  const [rows, setRows] = useState([]);

  useEffect(() => {
    fetchHitl().then((res) => setRows(res.data));
  }, []);

  const handleSave = async () => {
    await saveHitl(rows);
    alert("✅ HITL edits saved!");
  };

  return (
    <div className="p-10 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-4">🧍 Human Review (HITL)</h1>
      <div style={{ height: 600, width: "100%" }}>
        <DataGrid
          rows={rows}
          columns={[
            { field: "id", headerName: "ID", width: 80 },
            { field: "description", headerName: "Description", width: 300, editable: true },
            { field: "mapped_category", headerName: "Category", width: 200, editable: true },
            { field: "final_confidence", headerName: "Confidence", width: 150 },
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
      <button
        onClick={handleSave}
        className="mt-4 bg-blue-600 text-white px-5 py-2 rounded hover:bg-blue-700"
      >
        Save Changes
      </button>
    </div>
  );
}
