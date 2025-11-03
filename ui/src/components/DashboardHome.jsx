import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { FaBox, FaCheckCircle, FaExclamationCircle, FaPercent } from "react-icons/fa";

export default function DashboardHome() {
  const [stats, setStats] = useState({
    total: 0,
    approved: 0,
    pending: 0,
    avg_conf: 0,
  });

  useEffect(() => {
    // TODO: replace with API call (stats endpoint)
    setStats({ total: 1250, approved: 1025, pending: 225, avg_conf: 91.3 });
  }, []);

  const data = [
    { name: "Approved", value: stats.approved },
    { name: "Pending", value: stats.pending },
  ];

  const cards = [
    {
      title: "Total Records",
      value: stats.total,
      icon: <FaBox />,
      color: "bg-indigo-600",
    },
    {
      title: "Approved Records",
      value: stats.approved,
      icon: <FaCheckCircle />,
      color: "bg-emerald-600",
    },
    {
      title: "Pending HITL",
      value: stats.pending,
      icon: <FaExclamationCircle />,
      color: "bg-amber-500",
    },
    {
      title: "Avg Confidence",
      value: `${stats.avg_conf}%`,
      icon: <FaPercent />,
      color: "bg-sky-500",
    },
  ];

  return (
    <motion.div
      className="min-h-screen bg-gray-50 p-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Header */}
      <div className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">
            Inventory Standardization Dashboard
          </h1>
          <p className="text-gray-500">AI-Driven SKU Mapping and Review System</p>
        </div>
        <img src="/logo192.png" alt="logo" className="w-14 h-14" />
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {cards.map((card, i) => (
          <motion.div
            key={i}
            whileHover={{ scale: 1.05 }}
            className={`${card.color} text-white rounded-xl p-6 shadow-md flex items-center justify-between`}
          >
            <div>
              <h2 className="text-lg font-semibold">{card.title}</h2>
              <p className="text-2xl font-bold mt-2">{card.value}</p>
            </div>
            <div className="text-4xl opacity-80">{card.icon}</div>
          </motion.div>
        ))}
      </div>

      {/* Chart */}
      <div className="bg-white p-6 rounded-xl shadow-md">
        <h2 className="text-xl font-semibold mb-4 text-gray-700">Record Summary</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#4F46E5" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
