import { generateSku } from "../api/inventory_api";

export default function SkuGenerator() {
  const handleGenerate = async () => {
    await generateSku();
    alert("✅ SKU generation completed!");
  };

  return (
    <div className="p-10 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">🔖 SKU Generation</h1>
      <button
        onClick={handleGenerate}
        className="bg-purple-600 text-white px-5 py-2 rounded hover:bg-purple-700"
      >
        Generate SKUs
      </button>
    </div>
  );
}
