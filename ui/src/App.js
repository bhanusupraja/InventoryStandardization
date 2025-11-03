import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";

import DashboardHome from "./components/DashboardHome";
import FileUpload from "./components/FileUpload";
import InventoryTable from "./components/InventoryTable";
import HitlEditor from "./components/HitlEditor";
import SkuGenerator from "./components/SkuGenerator";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<DashboardHome />} />
          <Route path="/upload" element={<FileUpload />} />
          <Route path="/review" element={<InventoryTable />} />
          <Route path="/hitl" element={<HitlEditor />} />
          <Route path="/sku" element={<SkuGenerator />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
