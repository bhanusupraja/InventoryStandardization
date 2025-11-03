import { useNavigate, Outlet, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { useState } from "react";
import { FaUpload, FaTable, FaUserCheck, FaTags } from "react-icons/fa";

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(true);

  const menuItems = [
    { title: "Upload", icon: <FaUpload />, path: "/upload" },
    { title: "Review Records", icon: <FaTable />, path: "/review" },
    { title: "HITL Review", icon: <FaUserCheck />, path: "/hitl" },
    { title: "SKU Generation", icon: <FaTags />, path: "/sku" },
  ];

  return (
    <div className="flex h-screen bg-gray-100 font-inter">
      {/* Sidebar */}
      <motion.div
        animate={{ width: open ? 240 : 80 }}
        className="bg-indigo-900 text-white flex flex-col justify-between shadow-xl"
      >
        <div>
          {/* Logo + Title */}
          <div className="flex items-center gap-2 p-4 cursor-pointer" onClick={() => navigate("/")}>
            <img src="/logo192.png" alt="logo" className="w-8 h-8" />
            {open && <h1 className="text-lg font-bold leading-tight">Inventory Standardization</h1>}
          </div>

          {/* Nav Links */}
          <nav className="mt-6 flex flex-col gap-2">
            {menuItems.map((item, i) => (
              <div
                key={i}
                onClick={() => navigate(item.path)}
                className={`flex items-center gap-3 px-4 py-3 rounded-md cursor-pointer transition-all ${
                  location.pathname === item.path
                    ? "bg-indigo-700 text-white"
                    : "hover:bg-indigo-800 hover:text-gray-200"
                }`}
              >
                {item.icon}
                {open && <span>{item.title}</span>}
              </div>
            ))}
          </nav>
        </div>

        <div
          className="text-center py-4 text-gray-300 cursor-pointer border-t border-indigo-800"
          onClick={() => setOpen(!open)}
        >
          {open ? "◀" : "▶"}
        </div>
      </motion.div>

      {/* Main */}
      <div className="flex-1 flex flex-col">
        <header className="bg-white shadow flex justify-between items-center px-6 py-3">
          <h2 className="text-2xl font-semibold text-gray-700">
            {menuItems.find((m) => m.path === location.pathname)?.title || "Dashboard"}
          </h2>
          <div className="text-sm text-gray-500">v1.0 | POS Inventory</div>
        </header>

        <main className="flex-1 overflow-y-auto bg-gray-50 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
