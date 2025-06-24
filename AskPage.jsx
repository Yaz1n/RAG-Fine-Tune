import React, { useState } from "react";

const AskPage = () => {
  const [question, setQuestion] = useState("");
  const [teacherAnswer, setTeacherAnswer] = useState("");
  const [studentAnswer, setStudentAnswer] = useState("");
  const [context, setContext] = useState("");

  const handleSubmit = () => {
    if (!question.trim()) return;

    // Simulated Teacher and Student model outputs
    setTeacherAnswer("The mitochondria is the powerhouse of the cell.");
    setStudentAnswer("Mitochondria help provide energy to the cell.");
    setContext(
      "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. This energy is stored in a molecule called ATP (adenosine triphosphate)."
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 px-6 py-10 flex flex-col items-center">
      <h1 className="text-3xl font-bold text-indigo-800 mb-6">Ask the Models</h1>

      <input
        type="text"
        placeholder="Enter your question..."
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

      {(teacherAnswer || studentAnswer) && (
        <div className="bg-white shadow-md rounded-lg p-6 w-full max-w-xl space-y-6">
          <div>
            <h2 className="text-lg font-semibold text-indigo-700 mb-2">Teacher Model Answer:</h2>
            <p className="text-gray-800">{teacherAnswer}</p>
          </div>

          <div>
            <h2 className="text-lg font-semibold text-purple-700 mb-2">Student Model Answer:</h2>
            <p className="text-gray-800">{studentAnswer}</p>
          </div>

          <div>
            <h3 className="text-md font-semibold text-gray-700 mb-1">Context Paragraph:</h3>
            <p className="text-gray-600">{context}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AskPage;
