import React from "react";
import { Link } from "react-router-dom";


const Navbar = () => {
  return (
    <nav className="bg-indigo-700 text-white px-6 py-4 flex justify-between">
      <h1 className="text-xl font-bold">RAG FINE-TUNING</h1>
      <ul className="flex space-x-6">
        <li>
          <Link to="/" className="bg-white text-indigo-700 px-4 py-2 rounded-md font-medium hover:bg-indigo-200 transition">Home</Link>
        </li>
        <li>
        <Link to="/student-training" className="bg-white text-indigo-700 px-4 py-2 rounded-md font-medium hover:bg-indigo-200 transition">Student Training</Link>
        </li>
        <li>
          <Link to="/ask" className="bg-white text-indigo-700 px-4 py-2 rounded-md font-medium hover:bg-indigo-200 transition">Evaluation</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
