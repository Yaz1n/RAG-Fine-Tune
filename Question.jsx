import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const Question = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const dataset = location.state?.dataset || [];

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState("");

  const handleSubmit = () => {
    if (!question.trim() || dataset.length === 0) return;

    const random = dataset[Math.floor(Math.random() * dataset.length)];
    setAnswer(random.answer);
    setContext(random.context || "No context provided.");
  };

  if (dataset.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center text-red-600 font-semibold text-xl">
        No dataset found. Please <span className="ml-2 underline cursor-pointer text-blue-600" onClick={() => navigate("/")}>upload a file first</span>.
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 px-6 py-10 flex flex-col items-center">
      <h1 className="text-3xl font-bold text-indigo-800 mb-6">Ask a Question</h1>

      <input
        type="text"
        placeholder="Ask something..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
        className="w-full max-w-xl px-4 py-2 border border-indigo-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 mb-4"
      />

      <button
        onClick={handleSubmit}
        className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition mb-6"
      >
        Ask
      </button>

      {answer && (
        <div className="bg-white shadow-md rounded-lg p-6 w-full max-w-xl space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-indigo-700 mb-1">Answer:</h2>
            <p className="text-gray-800">{answer}</p>
          </div>

          <div>
            <h3 className="text-md font-semibold text-gray-700 mb-1">Context:</h3>
            <p className="text-gray-600">{context}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Question;
