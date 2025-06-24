import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import UploadPage from "./components/UploadPage";
import AskPage from "./components/AskPage";
import StudentTraining from "./components/StudentTraining";
import Question from "./components/Question";


function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<UploadPage />} />     
        <Route path="/ask" element={<AskPage />} />
        <Route path="/question" element={<Question  />} />
        <Route path="/student-training" element={<StudentTraining />} /> {/* 👈 new route */}
      </Routes>
    </Router>
  );
}

export default App;
