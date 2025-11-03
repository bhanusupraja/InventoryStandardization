import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const steps = [
  { title: "Upload File", path: "/upload" },
  { title: "Preprocessing", path: "/clean" },
  { title: "HITL Review", path: "/hitl" },
  { title: "SKU Generation", path: "/sku" },
  { title: "Final Inventory", path: "/final" },
];

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-50 p-10 text-center">
      <motion.h1
        className="text-4xl font-bold mb-8 text-gray-800"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        🧠 Inventory Standardization Dashboard
      </motion.h1>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
        {steps.map((step, i) => (
          <motion.div
            key={i}
            whileHover={{ scale: 1.05 }}
            className="bg-white shadow-md p-6 rounded-xl cursor-pointer hover:shadow-xl"
            onClick={() => navigate(step.path)}
          >
            <h2 className="text-xl font-semibold text-blue-600">{step.title}</h2>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
