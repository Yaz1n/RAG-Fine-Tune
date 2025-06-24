import React, { useState } from "react";

const StudentTraining = () => {
  const [loading, setLoading] = useState(false);
  const [generated, setGenerated] = useState(false);

  const handleGenerate = () => {
    setLoading(true);
    setGenerated(false);

    setTimeout(() => {
      setLoading(false);
      setGenerated(true);
    }, 2000); // Simulate generation time
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-100 flex flex-col items-center justify-center px-4">
      <h1 className="text-3xl font-bold text-green-800 mb-6">Generate Student Training Dataset</h1>

      <button
        onClick={handleGenerate}
        disabled={loading}
        className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition disabled:opacity-50"
      >
        {loading ? "Generating..." : "Generate"}
      </button>

      {generated && (
        <div className="mt-6 bg-white shadow-md rounded-lg p-6 w-full max-w-lg text-center">
          <h2 className="text-lg font-semibold text-green-700 mb-2">Success!</h2>
          <p className="text-gray-800 mb-4">Training dataset generated successfully.</p>

          <div className="text-left text-sm text-gray-700">
            <p><strong>Total Questions:</strong> 1500</p>
            <p><strong>Format:</strong> JSON</p>
            <p><strong>Model:</strong> Student-T5-small</p>
            <p><strong>Timestamp:</strong> {new Date().toLocaleString()}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default StudentTraining;
